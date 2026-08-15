# ============================================================
# MNIST SNN with MTJ-inspired nonlinear neuron
#
# Architecture:
#
# 784 -> 128 -> 64 -> 10
#
# Main changes compared with the previous version:
#
# 1. No hard "input_current > 0" gate during training.
# 2. Neural input is smoothly mapped to pulse strength.
# 3. MTJ nonlinear integration is preserved.
# 4. MTJ nonlinear leakage is preserved.
# 5. No hard reset after firing.
# 6. Classification uses output spike count.
# 7. Layer firing rates are monitored.
#
# ============================================================


import torch
import torch.nn as nn
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


# ============================================================
# MTJ neuron parameters
# ============================================================

# Start with 0.8 rather than 0.9.
# You can test 0.7, 0.8, and 0.9 later.
THRESHOLD = 0.8


# Maximum pulse width.
# Effective pulse width will be:
#
# pulse_width * drive
#
PULSE_WIDTH_PS = 30.0


# Simulation interval represented by one neural time step
DT_PS = 100.0


# Leak constant from the supplied model
TAU_LEAK_PS = 503.8


# Physical current range
J_MIN = 1e11
J_MAX = 1e12


# ============================================================
# Parametric sigmoid ONLY for integrate/leak selection
# ============================================================

GATE_SIGMOID_SLOPE = 1.0
GATE_SIGMOID_THRESHOLD = 0.0


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
# MNIST
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
# Surrogate gradient
# ============================================================

spike_grad = surrogate.fast_sigmoid(
    slope=25
)


# ============================================================
# Custom MTJ neuron
# ============================================================

class MTJNeuron(nn.Module):

    def __init__(
        self,
        threshold=0.8,
        pulse_width_ps=30.0,
        dt_ps=100.0,
        tau_leak_ps=503.8,
        J_min=1e11,
        J_max=1e12,
        input_scale=1.0
    ):

        super().__init__()


        self.threshold = threshold

        self.pulse_width_ps = pulse_width_ps

        self.dt_ps = dt_ps

        self.tau_leak_ps = tau_leak_ps

        self.J_min = J_min

        self.J_max = J_max


        # Controls sharpness of neural input -> physical drive
        self.input_scale = input_scale


    # ========================================================
    # A(J)
    # ========================================================

    def get_A(self, J):

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

            * 9.0
        )


        index_low = torch.floor(
            position
        ).long()


        index_high = torch.clamp(
            index_low + 1,
            max=9
        )


        alpha = (
            position
            -
            index_low.float()
        )


        A_low = A_values[index_low]

        A_high = A_values[index_high]


        A = (

            A_low
            *
            (1.0 - alpha)

            +

            A_high
            *
            alpha
        )


        return A


    # ========================================================
    # Current-dependent rise time
    # ========================================================

    def get_tau_r(self, J):

        J = torch.clamp(
            J,
            min=self.J_min,
            max=self.J_max
        )


        # Equation from the supplied model:
        #
        # tau_r =
        # 386.98 * (J/1e11)^(-1.223) + 8.88

        normalized_current = (
            J / 1e11
        )


        tau_r = (

            386.98

            *

            normalized_current.pow(
                -1.223
            )

            +

            8.88
        )


        return tau_r


    # ========================================================
    # Neural input -> normalized drive
    # ========================================================

    def get_drive(self, synaptic_input):

        # Smooth differentiable mapping
        #
        # drive is in:
        #
        # 0 <= drive <= 1

        drive = torch.sigmoid(

            self.input_scale
            *
            synaptic_input
        )


        return drive


    # ========================================================
    # Parametric sigmoid for integrate/leak selection ONLY
    # ========================================================

    def get_integrate_leak_gate(self, synaptic_input):

        # gate = sigmoid(k * (z - z0))
        #
        # Larger k -> closer to a hard threshold:
        #   z > z0  -> gate ~ 1 -> integrate
        #   z < z0  -> gate ~ 0 -> leak
        #
        # This gate does NOT affect the drive, current density,
        # or effective pulse width.

        gate = torch.sigmoid(
            GATE_SIGMOID_SLOPE
            *
            (
                synaptic_input
                -
                GATE_SIGMOID_THRESHOLD
            )
        )

        return gate


    # ========================================================
    # Drive -> current density
    # ========================================================

    def drive_to_current(
        self,
        drive
    ):

        J = (

            self.J_min

            +

            drive

            *

            (
                self.J_max
                -
                self.J_min
            )
        )


        return J


    # ========================================================
    # MTJ integration equation
    # ========================================================

    def integrate(
        self,
        mem,
        drive
    ):

        # ----------------------------------------------------
        # Convert normalized drive to physical J
        # ----------------------------------------------------

        J = self.drive_to_current(
            drive
        )


        # ----------------------------------------------------
        # Get physical neuron parameters
        # ----------------------------------------------------

        A = self.get_A(
            J
        )


        tau_r = self.get_tau_r(
            J
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Instead of applying the full pulse width whenever
        # synaptic input is positive, pulse strength is scaled
        # continuously by drive.
        #
        # effective pulse:
        #
        # T_eff = drive * T_pulse
        # ----------------------------------------------------

        effective_pulse_width = (

            drive
            *
            self.pulse_width_ps
        )


        # ----------------------------------------------------
        # Prevent atanh domain errors
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MTJ nonlinear integration
        # ----------------------------------------------------

        mem_integrated = (

            A

            *

            torch.tanh(

                effective_pulse_width
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
    # MTJ nonlinear leakage equation
    # ========================================================

    def leak(
        self,
        mem
    ):

        decay_value = (

            -self.dt_ps
            /
            self.tau_leak_ps
        )


        B = torch.exp(

            torch.tensor(
                decay_value,
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
        synaptic_input,
        mem
    ):

        # ----------------------------------------------------
        # Convert FC output into normalized physical drive
        # ----------------------------------------------------

        drive = self.get_drive(
            synaptic_input
        )


        # ----------------------------------------------------
        # Parametric sigmoid ONLY for integrate/leak selection
        # ----------------------------------------------------

        integrate_leak_gate = self.get_integrate_leak_gate(
            synaptic_input
        )


        # ----------------------------------------------------
        # Compute integration
        # ----------------------------------------------------

        integrated_mem = self.integrate(
            mem,
            drive
        )


        # ----------------------------------------------------
        # Compute leak-only dynamics
        # ----------------------------------------------------

        leaked_mem = self.leak(
            mem
        )


        # ----------------------------------------------------
        # Smooth combination
        #
        # integrate_leak_gate = 1
        # -> mostly integration
        #
        # integrate_leak_gate = 0
        # -> mostly leakage
        #
        # The gate uses a parametric sigmoid, while the
        # original drive sigmoid remains completely unchanged.
        # ----------------------------------------------------

        mem_new = (

            integrate_leak_gate
            *
            integrated_mem

            +

            (
                1.0
                -
                integrate_leak_gate
            )

            *
            leaked_mem
        )


        # ----------------------------------------------------
        # Keep physical state bounded
        # ----------------------------------------------------

        mem_new = torch.clamp(
            mem_new,
            min=0.0,
            max=0.999
        )


        # ----------------------------------------------------
        # Spike generation
        # ----------------------------------------------------

        spk = spike_grad(

            mem_new
            -
            self.threshold
        )


        # ----------------------------------------------------
        # NO hard reset
        # ----------------------------------------------------
        #
        # The supplied model does not explicitly define
        # reset-to-zero after firing.
        #
        # Therefore membrane state is preserved.
        # ----------------------------------------------------

        return (
            spk,
            mem_new
        )


# ============================================================
# Full Spiking MLP
# ============================================================

class MTJSpikingMLP(nn.Module):

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


        self.neuron1 = MTJNeuron(

            threshold=THRESHOLD,

            pulse_width_ps=PULSE_WIDTH_PS,

            dt_ps=DT_PS,

            tau_leak_ps=TAU_LEAK_PS,

            J_min=J_MIN,

            J_max=J_MAX,

            input_scale=1.0
        )


        # ====================================================
        # Layer 2
        # ====================================================

        self.fc2 = nn.Linear(
            128,
            64
        )


        self.neuron2 = MTJNeuron(

            threshold=THRESHOLD,

            pulse_width_ps=PULSE_WIDTH_PS,

            dt_ps=DT_PS,

            tau_leak_ps=TAU_LEAK_PS,

            J_min=J_MIN,

            J_max=J_MAX,

            input_scale=1.0
        )


        # ====================================================
        # Layer 3
        # ====================================================

        self.fc3 = nn.Linear(
            64,
            10
        )


        self.neuron3 = MTJNeuron(

            threshold=THRESHOLD,

            pulse_width_ps=PULSE_WIDTH_PS,

            dt_ps=DT_PS,

            tau_leak_ps=TAU_LEAK_PS,

            J_min=J_MIN,

            J_max=J_MAX,

            input_scale=1.0
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        return_layer_spikes=False
    ):

        batch_size = x.shape[0]


        x = self.flatten(
            x
        )


        # ====================================================
        # Initial MTJ states
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
        # Recording
        # ====================================================

        spk1_record = []

        spk2_record = []

        spk3_record = []


        mem3_record = []


        # ====================================================
        # Temporal simulation
        # ====================================================

        for t in range(
            self.num_steps
        ):


            # =================================================
            # First layer
            # =================================================

            current1 = self.fc1(
                x
            )


            spk1, mem1 = self.neuron1(

                current1,

                mem1
            )


            # =================================================
            # Second layer
            # =================================================

            current2 = self.fc2(
                spk1
            )


            spk2, mem2 = self.neuron2(

                current2,

                mem2
            )


            # =================================================
            # Output layer
            # =================================================

            current3 = self.fc3(
                spk2
            )


            spk3, mem3 = self.neuron3(

                current3,

                mem3
            )


            # =================================================
            # Record
            # =================================================

            spk1_record.append(
                spk1
            )


            spk2_record.append(
                spk2
            )


            spk3_record.append(
                spk3
            )


            mem3_record.append(
                mem3
            )


        # ====================================================
        # Convert to:
        #
        # [time, batch, neuron]
        # ====================================================

        spk1_record = torch.stack(
            spk1_record
        )


        spk2_record = torch.stack(
            spk2_record
        )


        spk3_record = torch.stack(
            spk3_record
        )


        mem3_record = torch.stack(
            mem3_record
        )


        if return_layer_spikes:

            return (

                spk1_record,

                spk2_record,

                spk3_record,

                mem3_record
            )


        return (
            spk3_record,
            mem3_record
        )


# ============================================================
# Create model
# ============================================================

model = MTJSpikingMLP(
    num_steps=NUM_STEPS
).to(
    device
)


print("\nNetwork:")
print(model)


# ============================================================
# Number of parameters
# ============================================================

trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)


print(
    "\nTrainable parameters:",
    trainable_parameters
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
# Gradient clipping
# ============================================================

GRAD_CLIP = 1.0


# ============================================================
# Training function
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


    firing_rate_1 = 0.0

    firing_rate_2 = 0.0

    firing_rate_3 = 0.0


    number_batches = 0


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
        # Forward pass
        # ====================================================

        (

            spk1,

            spk2,

            spk3,

            mem3

        ) = model(

            images,

            return_layer_spikes=True
        )


        # ====================================================
        # Spike-count decoding
        # ====================================================
        #
        # Shape:
        #
        # spk3:
        # [T, B, 10]
        #
        # output:
        # [B, 10]
        # ====================================================

        output = spk3.sum(
            dim=0
        )


        # ====================================================
        # Classification loss
        # ====================================================

        loss = criterion(
            output,
            labels
        )


        # ====================================================
        # Backpropagation Through Time
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
        # Firing rates
        # ====================================================

        firing_rate_1 += (
            spk1.detach().mean().item()
        )


        firing_rate_2 += (
            spk2.detach().mean().item()
        )


        firing_rate_3 += (
            spk3.detach().mean().item()
        )


        number_batches += 1


        # ====================================================
        # Print intermediate status
        # ====================================================

        if batch_idx % 100 == 0:

            print(

                f"Epoch {epoch} | "

                f"Batch {batch_idx:4d}/{len(loader)} | "

                f"Loss {loss.item():.4f} | "

                f"FR1 {spk1.mean().item():.4f} | "

                f"FR2 {spk2.mean().item():.4f} | "

                f"FR3 {spk3.mean().item():.4f}"
            )


    # ========================================================
    # Epoch results
    # ========================================================

    average_loss = (

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


    firing_rate_1 /= number_batches

    firing_rate_2 /= number_batches

    firing_rate_3 /= number_batches


    return (

        average_loss,

        accuracy,

        firing_rate_1,

        firing_rate_2,

        firing_rate_3
    )


# ============================================================
# Test function
# ============================================================

def evaluate(

    model,
    loader,
    device

):


    model.eval()


    correct = 0

    total = 0


    firing_rate_1 = 0.0

    firing_rate_2 = 0.0

    firing_rate_3 = 0.0


    number_batches = 0


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

                mem3

            ) = model(

                images,

                return_layer_spikes=True
            )


            # =================================================
            # Spike-count decoding
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


            firing_rate_1 += (
                spk1.mean().item()
            )


            firing_rate_2 += (
                spk2.mean().item()
            )


            firing_rate_3 += (
                spk3.mean().item()
            )


            number_batches += 1


    accuracy = (

        100.0
        *
        correct
        /
        total
    )


    firing_rate_1 /= number_batches

    firing_rate_2 /= number_batches

    firing_rate_3 /= number_batches


    return (

        accuracy,

        firing_rate_1,

        firing_rate_2,

        firing_rate_3
    )


# ============================================================
# Training
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


    (

        train_loss,

        train_accuracy,

        train_fr1,

        train_fr2,

        train_fr3

    ) = train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        device,

        epoch
    )


    (

        test_accuracy,

        test_fr1,

        test_fr2,

        test_fr3

    ) = evaluate(

        model,

        test_loader,

        device
    )


    print("\n---------------- Results ----------------")

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy:    {test_accuracy:.2f}%"
    )


    print("\nTraining firing rates:")

    print(
        f"Layer 1: {train_fr1:.4f}"
    )

    print(
        f"Layer 2: {train_fr2:.4f}"
    )

    print(
        f"Layer 3: {train_fr3:.4f}"
    )


    print("\nTest firing rates:")

    print(
        f"Layer 1: {test_fr1:.4f}"
    )

    print(
        f"Layer 2: {test_fr2:.4f}"
    )

    print(
        f"Layer 3: {test_fr3:.4f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if test_accuracy > best_accuracy:


        best_accuracy = test_accuracy


        torch.save(

            model.state_dict(),

            "best_mtjlif_mnist.pth"
        )


        print(
            "\nBest model saved."
        )


# ============================================================
# Final result
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
# Reload best model
# ============================================================

model.load_state_dict(

    torch.load(

        "best_mtjlif_mnist.pth",

        map_location=device
    )
)


# ============================================================
# Final evaluation
# ============================================================

(

    final_accuracy,

    final_fr1,

    final_fr2,

    final_fr3

) = evaluate(

    model,

    test_loader,

    device
)


print(
    f"\nFinal Test Accuracy: {final_accuracy:.2f}%"
)


print(
    "\nFinal firing rates:"
)

print(
    f"Layer 1 = {final_fr1:.4f}"
)

print(
    f"Layer 2 = {final_fr2:.4f}"
)

print(
    f"Layer 3 = {final_fr3:.4f}"
)