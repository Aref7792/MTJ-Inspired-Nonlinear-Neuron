# ============================================================
# MNIST Spiking MLP with HARD MTJ pulse/no-pulse dynamics
#
# Architecture:
# 784 -> 128 -> 64 -> 10
#
# Forward pass:
#
# z > 0  -> HARD pulse = 1 -> MTJ integrate
# z <= 0 -> HARD pulse = 0 -> MTJ leak
#
# Backward pass:
# sigmoid-based straight-through gradient
#
# Output:
# spike-count classification
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snntorch import surrogate


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_STEPS = 25

# Spike threshold for m_z
THRESHOLD = 0.8

# Fixed physical pulse width
PULSE_WIDTH_PS = 30.0

# Leak interval
DT_PS = 100.0

# Leak time constant
TAU_LEAK_PS = 503.8

# Characterized current-density range
J_MIN = 1e11
J_MAX = 1e12

# Controls steepness of surrogate pulse gate
PULSE_SURROGATE_ALPHA = 5.0

# Gradient clipping
GRAD_CLIP = 1.0


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


print("Using device:", device)


# ============================================================
# MNIST Dataset
# ============================================================

transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Surrogate gradient for OUTPUT SPIKE
# ============================================================

output_spike_grad = surrogate.fast_sigmoid(
    slope=25
)


# ============================================================
# Hard pulse gate with straight-through gradient
# ============================================================

def hard_pulse_ste(z, alpha=5.0):
    """
    Forward:
        pulse = 1 if z > 0
        pulse = 0 otherwise

    Backward:
        gradient approximated by sigmoid(alpha*z)

    This gives physically hard pulse/no-pulse behavior
    while still allowing training.
    """

    # Soft value used only for backward gradient
    soft = torch.sigmoid(alpha * z)

    # Hard physical decision
    hard = (z > 0).to(z.dtype)

    # Straight-through estimator
    #
    # Forward:
    # hard + soft - soft = hard
    #
    # Backward:
    # gradient flows through soft
    pulse = hard + soft - soft.detach()

    return pulse


# ============================================================
# MTJ Neuron
# ============================================================

class HardMTJNeuron(nn.Module):

    def __init__(
        self,
        threshold=0.8,
        pulse_width_ps=30.0,
        dt_ps=100.0,
        tau_leak_ps=503.8,
        J_min=1e11,
        J_max=1e12,
        pulse_alpha=5.0
    ):

        super().__init__()

        self.threshold = threshold

        self.pulse_width_ps = pulse_width_ps

        self.dt_ps = dt_ps

        self.tau_leak_ps = tau_leak_ps

        self.J_min = J_min

        self.J_max = J_max

        self.pulse_alpha = pulse_alpha


    # ========================================================
    # A(J)
    # ========================================================

    def get_A(self, J):

        # Values from the supplied characterization table

        J_values = torch.tensor(
            [
                1e11,
                2e11,
                3e11,
                4e11,
                5e11,
                6e11,
                7e11,
                8e11,
                9e11,
                1e12
            ],
            device=J.device,
            dtype=J.dtype
        )


        A_values = torch.tensor(
            [
                0.9855,
                0.9991,
                0.9997,
                0.9998,
                0.9999,
                0.99993,
                0.99999,
                0.9999,
                1.0,
                1.0
            ],
            device=J.device,
            dtype=J.dtype
        )


        J = torch.clamp(
            J,
            min=self.J_min,
            max=self.J_max
        )


        position = (
            (J - self.J_min)
            /
            (self.J_max - self.J_min)
            *
            9.0
        )


        idx_low = torch.floor(
            position
        ).long()


        idx_high = torch.clamp(
            idx_low + 1,
            max=9
        )


        alpha = (
            position
            -
            idx_low.float()
        )


        A_low = A_values[idx_low]

        A_high = A_values[idx_high]


        A = (
            A_low * (1.0 - alpha)
            +
            A_high * alpha
        )


        return A


    # ========================================================
    # Rise-time equation
    # ========================================================

    def get_tau_r(self, J):

        J = torch.clamp(
            J,
            min=self.J_min,
            max=self.J_max
        )


        normalized_J = (
            J / 1e11
        )


        tau_r = (
            386.98
            *
            normalized_J.pow(-1.223)
            +
            8.88
        )


        return tau_r


    # ========================================================
    # Positive z magnitude -> pulse current density
    # ========================================================

    def map_z_to_current(self, z):
        """
        Pulse existence is determined separately by z > 0.

        Here, positive z magnitude controls J.

        We use:
            strength = 1 - exp(-ReLU(z))

        Therefore:
            z <= 0 -> strength = 0
            small positive z -> small strength
            large positive z -> strength -> 1

        Then:
            J = J_min + strength * (J_max - J_min)

        Note:
        this neural-to-current mapping is a modeling/interface
        assumption, not a device equation from the slides.
        """

        positive_z = F.relu(z)

        strength = (
            1.0
            -
            torch.exp(
                -positive_z
            )
        )


        J = (
            self.J_min
            +
            strength
            *
            (
                self.J_max
                -
                self.J_min
            )
        )


        return J


    # ========================================================
    # MTJ Integration
    # ========================================================

    def integrate(self, mem, J):

        A = self.get_A(J)

        tau_r = self.get_tau_r(J)


        ratio = (
            mem
            /
            (A + 1e-8)
        )


        ratio = torch.clamp(
            ratio,
            min=-0.999,
            max=0.999
        )


        mem_integrated = (
            A
            *
            torch.tanh(
                self.pulse_width_ps
                /
                (tau_r + 1e-8)
                +
                torch.atanh(
                    ratio
                )
            )
        )


        return mem_integrated


    # ========================================================
    # MTJ Leakage
    # ========================================================

    def leak(self, mem):

        decay = (
            -self.dt_ps
            /
            self.tau_leak_ps
        )


        B = torch.exp(
            torch.tensor(
                decay,
                device=mem.device,
                dtype=mem.dtype
            )
        )


        denominator = torch.sqrt(
            torch.clamp(
                1.0
                -
                mem.pow(2)
                *
                (
                    1.0
                    -
                    B.pow(2)
                ),
                min=1e-8
            )
        )


        mem_leaked = (
            mem
            *
            B
            /
            denominator
        )


        return mem_leaked


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        z,
        mem
    ):

        # ----------------------------------------------------
        # HARD pulse existence
        # ----------------------------------------------------
        #
        # Forward:
        #
        # z > 0  -> 1
        # z <= 0 -> 0
        #
        # Backward:
        # sigmoid surrogate
        # ----------------------------------------------------

        pulse = hard_pulse_ste(
            z,
            alpha=self.pulse_alpha
        )


        # ----------------------------------------------------
        # Determine physical current density
        # ----------------------------------------------------

        J = self.map_z_to_current(
            z
        )


        # ----------------------------------------------------
        # Candidate if pulse exists
        # ----------------------------------------------------

        mem_integrated = self.integrate(
            mem,
            J
        )


        # ----------------------------------------------------
        # Candidate if no pulse
        # ----------------------------------------------------

        mem_leaked = self.leak(
            mem
        )


        # ----------------------------------------------------
        # HARD physical selection
        # ----------------------------------------------------
        #
        # Because pulse is exactly 0 or 1 in forward pass:
        #
        # pulse = 1:
        #     mem_new = mem_integrated
        #
        # pulse = 0:
        #     mem_new = mem_leaked
        #
        # ----------------------------------------------------

        mem_new = (
            pulse
            *
            mem_integrated
            +
            (
                1.0
                -
                pulse
            )
            *
            mem_leaked
        )


        # Keep physical state in valid region
        mem_new = torch.clamp(
            mem_new,
            min=0.0,
            max=0.999
        )


        # ----------------------------------------------------
        # Output spike
        # ----------------------------------------------------

        spk = output_spike_grad(
            mem_new
            -
            self.threshold
        )


        # ----------------------------------------------------
        # NO RESET
        # ----------------------------------------------------
        #
        # We do not reset m_z to zero after firing because
        # the supplied slides do not specify such a reset.
        # ----------------------------------------------------

        return (
            spk,
            mem_new,
            pulse
        )


# ============================================================
# Full MNIST Network
# ============================================================

class HardMTJSpikingMLP(nn.Module):

    def __init__(
        self,
        num_steps=25
    ):

        super().__init__()

        self.num_steps = num_steps

        self.flatten = nn.Flatten()


        # ====================================================
        # Layer 1
        # ====================================================

        self.fc1 = nn.Linear(
            784,
            128
        )


        self.neuron1 = HardMTJNeuron(
            threshold=THRESHOLD,
            pulse_width_ps=PULSE_WIDTH_PS,
            dt_ps=DT_PS,
            tau_leak_ps=TAU_LEAK_PS,
            J_min=J_MIN,
            J_max=J_MAX,
            pulse_alpha=PULSE_SURROGATE_ALPHA
        )


        # ====================================================
        # Layer 2
        # ====================================================

        self.fc2 = nn.Linear(
            128,
            64
        )


        self.neuron2 = HardMTJNeuron(
            threshold=THRESHOLD,
            pulse_width_ps=PULSE_WIDTH_PS,
            dt_ps=DT_PS,
            tau_leak_ps=TAU_LEAK_PS,
            J_min=J_MIN,
            J_max=J_MAX,
            pulse_alpha=PULSE_SURROGATE_ALPHA
        )


        # ====================================================
        # Output layer
        # ====================================================

        self.fc3 = nn.Linear(
            64,
            10
        )


        self.neuron3 = HardMTJNeuron(
            threshold=THRESHOLD,
            pulse_width_ps=PULSE_WIDTH_PS,
            dt_ps=DT_PS,
            tau_leak_ps=TAU_LEAK_PS,
            J_min=J_MIN,
            J_max=J_MAX,
            pulse_alpha=PULSE_SURROGATE_ALPHA
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        return_all=False
    ):

        batch_size = x.size(0)

        x = self.flatten(x)


        # ====================================================
        # Initial m_z
        # ====================================================

        mem1 = torch.zeros(
            batch_size,
            128,
            device=x.device,
            dtype=x.dtype
        )


        mem2 = torch.zeros(
            batch_size,
            64,
            device=x.device,
            dtype=x.dtype
        )


        mem3 = torch.zeros(
            batch_size,
            10,
            device=x.device,
            dtype=x.dtype
        )


        # ====================================================
        # Records
        # ====================================================

        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        pulse1_rec = []
        pulse2_rec = []
        pulse3_rec = []

        mem3_rec = []


        # ====================================================
        # Temporal simulation
        # ====================================================

        for t in range(
            self.num_steps
        ):


            # =================================================
            # Layer 1
            # =================================================

            z1 = self.fc1(
                x
            )


            spk1, mem1, pulse1 = self.neuron1(
                z1,
                mem1
            )


            # =================================================
            # Layer 2
            # =================================================

            z2 = self.fc2(
                spk1
            )


            spk2, mem2, pulse2 = self.neuron2(
                z2,
                mem2
            )


            # =================================================
            # Output Layer
            # =================================================

            z3 = self.fc3(
                spk2
            )


            spk3, mem3, pulse3 = self.neuron3(
                z3,
                mem3
            )


            # =================================================
            # Record
            # =================================================

            spk1_rec.append(
                spk1
            )

            spk2_rec.append(
                spk2
            )

            spk3_rec.append(
                spk3
            )


            pulse1_rec.append(
                pulse1
            )

            pulse2_rec.append(
                pulse2
            )

            pulse3_rec.append(
                pulse3
            )


            mem3_rec.append(
                mem3
            )


        # ====================================================
        # Stack
        # ====================================================

        spk1_rec = torch.stack(
            spk1_rec
        )

        spk2_rec = torch.stack(
            spk2_rec
        )

        spk3_rec = torch.stack(
            spk3_rec
        )


        pulse1_rec = torch.stack(
            pulse1_rec
        )

        pulse2_rec = torch.stack(
            pulse2_rec
        )

        pulse3_rec = torch.stack(
            pulse3_rec
        )


        mem3_rec = torch.stack(
            mem3_rec
        )


        if return_all:

            return (
                spk1_rec,
                spk2_rec,
                spk3_rec,
                pulse1_rec,
                pulse2_rec,
                pulse3_rec,
                mem3_rec
            )


        return (
            spk3_rec,
            mem3_rec
        )


# ============================================================
# Initialize Model
# ============================================================

model = HardMTJSpikingMLP(
    num_steps=NUM_STEPS
).to(
    device
)


print("\nModel:")
print(model)


# ============================================================
# Parameter Count
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    "\nTrainable parameters:",
    num_parameters
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch
):

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    fr1_total = 0.0
    fr2_total = 0.0
    fr3_total = 0.0


    pr1_total = 0.0
    pr2_total = 0.0
    pr3_total = 0.0


    num_batches = 0


    for batch_idx, (
        images,
        labels
    ) in enumerate(
        loader
    ):


        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        # ====================================================
        # Forward
        # ====================================================

        (
            spk1,
            spk2,
            spk3,
            pulse1,
            pulse2,
            pulse3,
            mem3

        ) = model(
            images,
            return_all=True
        )


        # ====================================================
        # Output spike counts
        # ====================================================

        output = spk3.sum(
            dim=0
        )


        # ====================================================
        # Cross-entropy
        # ====================================================

        loss = criterion(
            output,
            labels
        )


        # ====================================================
        # BPTT
        # ====================================================

        loss.backward()


        # ====================================================
        # Gradient clipping
        # ====================================================

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )


        optimizer.step()


        # ====================================================
        # Accuracy
        # ====================================================

        predictions = output.argmax(
            dim=1
        )


        correct += (
            predictions
            ==
            labels
        ).sum().item()


        total += labels.size(
            0
        )


        running_loss += loss.item()


        # ====================================================
        # Output spike firing rates
        # ====================================================

        fr1 = spk1.detach().mean().item()
        fr2 = spk2.detach().mean().item()
        fr3 = spk3.detach().mean().item()


        fr1_total += fr1
        fr2_total += fr2
        fr3_total += fr3


        # ====================================================
        # Input pulse rates
        # ====================================================

        pr1 = pulse1.detach().mean().item()
        pr2 = pulse2.detach().mean().item()
        pr3 = pulse3.detach().mean().item()


        pr1_total += pr1
        pr2_total += pr2
        pr3_total += pr3


        num_batches += 1


        # ====================================================
        # Status
        # ====================================================

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx:4d}/{len(loader)} | "
                f"Loss {loss.item():.4f} | "
                f"Spike FR: "
                f"{fr1:.3f}, "
                f"{fr2:.3f}, "
                f"{fr3:.3f} | "
                f"Pulse Rate: "
                f"{pr1:.3f}, "
                f"{pr2:.3f}, "
                f"{pr3:.3f}"
            )


    avg_loss = (
        running_loss
        /
        len(loader)
    )


    accuracy = (
        100.0
        *
        correct
        /
        total
    )


    return {
        "loss": avg_loss,

        "accuracy": accuracy,

        "fr1": fr1_total / num_batches,
        "fr2": fr2_total / num_batches,
        "fr3": fr3_total / num_batches,

        "pr1": pr1_total / num_batches,
        "pr2": pr2_total / num_batches,
        "pr3": pr3_total / num_batches,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()


    correct = 0
    total = 0


    fr1_total = 0.0
    fr2_total = 0.0
    fr3_total = 0.0


    pr1_total = 0.0
    pr2_total = 0.0
    pr3_total = 0.0


    num_batches = 0


    with torch.no_grad():


        for images, labels in loader:


            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            (
                spk1,
                spk2,
                spk3,
                pulse1,
                pulse2,
                pulse3,
                mem3

            ) = model(
                images,
                return_all=True
            )


            # =================================================
            # Spike count classification
            # =================================================

            output = spk3.sum(
                dim=0
            )


            predictions = output.argmax(
                dim=1
            )


            correct += (
                predictions
                ==
                labels
            ).sum().item()


            total += labels.size(
                0
            )


            # =================================================
            # Rates
            # =================================================

            fr1_total += (
                spk1.mean().item()
            )

            fr2_total += (
                spk2.mean().item()
            )

            fr3_total += (
                spk3.mean().item()
            )


            pr1_total += (
                pulse1.mean().item()
            )

            pr2_total += (
                pulse2.mean().item()
            )

            pr3_total += (
                pulse3.mean().item()
            )


            num_batches += 1


    accuracy = (
        100.0
        *
        correct
        /
        total
    )


    return {
        "accuracy": accuracy,

        "fr1": fr1_total / num_batches,
        "fr2": fr2_total / num_batches,
        "fr3": fr3_total / num_batches,

        "pr1": pr1_total / num_batches,
        "pr2": pr2_total / num_batches,
        "pr3": pr3_total / num_batches,
    }


# ============================================================
# Main Training Loop
# ============================================================

best_accuracy = 0.0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):


    print(
        "\n===================================================="
    )

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        "===================================================="
    )


    train_stats = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch
    )


    test_stats = evaluate(
        model,
        test_loader,
        device
    )


    print(
        "\n---------------- TRAIN ----------------"
    )

    print(
        f"Loss:           {train_stats['loss']:.4f}"
    )

    print(
        f"Accuracy:       {train_stats['accuracy']:.2f}%"
    )


    print(
        "\nOutput spike firing rates:"
    )

    print(
        f"Layer 1:        {train_stats['fr1']:.4f}"
    )

    print(
        f"Layer 2:        {train_stats['fr2']:.4f}"
    )

    print(
        f"Layer 3:        {train_stats['fr3']:.4f}"
    )


    print(
        "\nHard pulse rates:"
    )

    print(
        f"Layer 1:        {train_stats['pr1']:.4f}"
    )

    print(
        f"Layer 2:        {train_stats['pr2']:.4f}"
    )

    print(
        f"Layer 3:        {train_stats['pr3']:.4f}"
    )


    print(
        "\n---------------- TEST ----------------"
    )

    print(
        f"Accuracy:       {test_stats['accuracy']:.2f}%"
    )


    print(
        "\nOutput spike firing rates:"
    )

    print(
        f"Layer 1:        {test_stats['fr1']:.4f}"
    )

    print(
        f"Layer 2:        {test_stats['fr2']:.4f}"
    )

    print(
        f"Layer 3:        {test_stats['fr3']:.4f}"
    )


    print(
        "\nHard pulse rates:"
    )

    print(
        f"Layer 1:        {test_stats['pr1']:.4f}"
    )

    print(
        f"Layer 2:        {test_stats['pr2']:.4f}"
    )

    print(
        f"Layer 3:        {test_stats['pr3']:.4f}"
    )


    # ========================================================
    # Save best
    # ========================================================

    if test_stats["accuracy"] > best_accuracy:

        best_accuracy = test_stats["accuracy"]


        torch.save(
            model.state_dict(),
            "best_hard_mtjlif_mnist.pth"
        )


        print(
            "\nBest model saved."
        )


# ============================================================
# Final
# ============================================================

print(
    "\n===================================================="
)

print(
    "Training complete"
)

print(
    "===================================================="
)

print(
    f"Best Test Accuracy: {best_accuracy:.2f}%"
)


# ============================================================
# Load Best Model
# ============================================================

model.load_state_dict(
    torch.load(
        "best_hard_mtjlif_mnist.pth",
        map_location=device
    )
)


# ============================================================
# Final Test
# ============================================================

final_stats = evaluate(
    model,
    test_loader,
    device
)


print(
    f"\nFinal Test Accuracy: "
    f"{final_stats['accuracy']:.2f}%"
)


print(
    "\nFinal output firing rates:"
)

print(
    f"Layer 1 = {final_stats['fr1']:.4f}"
)

print(
    f"Layer 2 = {final_stats['fr2']:.4f}"
)

print(
    f"Layer 3 = {final_stats['fr3']:.4f}"
)


print(
    "\nFinal hard pulse rates:"
)

print(
    f"Layer 1 = {final_stats['pr1']:.4f}"
)

print(
    f"Layer 2 = {final_stats['pr2']:.4f}"
)

print(
    f"Layer 3 = {final_stats['pr3']:.4f}"
)