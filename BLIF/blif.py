# ============================================================
# Spiking version of:
# https://github.com/mytechnotalent/MNIST-MLP
#
# Original architecture:
# 784 -> 128 -> 64 -> 10
#
# Spiking architecture:
# 784 -> FC -> LIF -> FC -> LIF -> FC -> LIF
#
# Framework: snnTorch
#
# IMPORTANT:
# The neuron model is the standard snnTorch LIF neuron.
#
# This version only changes the logging/output format so that
# the results can be compared directly with the MTJ model.
# ============================================================


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
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

batch_size = 64

learning_rate = 0.01

num_epochs = 10


# Number of simulation time steps
num_steps = 25


# LIF membrane decay
beta = 0.9


# Surrogate gradient
spike_grad = surrogate.fast_sigmoid(
    slope=25
)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

    print("Using Apple MPS device.")


elif torch.cuda.is_available():

    device = torch.device("cuda")

    print("Using CUDA device.")


else:

    device = torch.device("cpu")

    print("Using CPU.")


# ============================================================
# MNIST Dataset
# ============================================================
#
# Same preprocessing as the original repo:
#
# only ToTensor()
#
# pixels remain in [0, 1]
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

    dataset=train_dataset,

    batch_size=batch_size,

    shuffle=True
)


test_loader = DataLoader(

    dataset=test_dataset,

    batch_size=batch_size,

    shuffle=False
)


# ============================================================
# Spiking MLP
# ============================================================

class SpikingMNISTModel(nn.Module):


    def __init__(self):

        super().__init__()


        # ====================================================
        # Flatten 28 x 28 -> 784
        # ====================================================

        self.flatten = nn.Flatten()


        # ====================================================
        # Layer 1
        #
        # 784 -> 128
        # ====================================================

        self.fc1 = nn.Linear(
            28 * 28,
            128
        )


        self.lif1 = snn.Leaky(

            beta=beta,

            spike_grad=spike_grad,

            init_hidden=False
        )


        # ====================================================
        # Layer 2
        #
        # 128 -> 64
        # ====================================================

        self.fc2 = nn.Linear(
            128,
            64
        )


        self.lif2 = snn.Leaky(

            beta=beta,

            spike_grad=spike_grad,

            init_hidden=False
        )


        # ====================================================
        # Layer 3
        #
        # 64 -> 10
        # ====================================================

        self.fc3 = nn.Linear(
            64,
            10
        )


        self.lif3 = snn.Leaky(

            beta=beta,

            spike_grad=spike_grad,

            init_hidden=False
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        return_all=False
    ):


        # ----------------------------------------------------
        # Input:
        #
        # [batch, 1, 28, 28]
        #
        # Flatten:
        #
        # [batch, 784]
        # ----------------------------------------------------

        x = self.flatten(
            x
        )


        # ====================================================
        # Initialize membrane potentials
        # ====================================================

        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()

        mem3 = self.lif3.init_leaky()


        # ====================================================
        # Record spikes from ALL layers
        #
        # Needed to calculate firing rates.
        # ====================================================

        spk1_rec = []

        spk2_rec = []

        spk3_rec = []


        # Output membrane recording
        mem3_rec = []


        # ====================================================
        # Temporal simulation
        # ====================================================

        for step in range(
            num_steps
        ):


            # =================================================
            # Layer 1
            # =================================================

            cur1 = self.fc1(
                x
            )


            spk1, mem1 = self.lif1(

                cur1,

                mem1
            )


            # =================================================
            # Layer 2
            # =================================================

            cur2 = self.fc2(
                spk1
            )


            spk2, mem2 = self.lif2(

                cur2,

                mem2
            )


            # =================================================
            # Layer 3
            # =================================================

            cur3 = self.fc3(
                spk2
            )


            spk3, mem3 = self.lif3(

                cur3,

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


            mem3_rec.append(
                mem3
            )


        # ====================================================
        # Convert lists to tensors
        #
        # Shape:
        #
        # [time, batch, neurons]
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


        mem3_rec = torch.stack(
            mem3_rec
        )


        # ====================================================
        # Return all layer spikes when requested
        # ====================================================

        if return_all:

            return (

                spk1_rec,

                spk2_rec,

                spk3_rec,

                mem3_rec
            )


        # Original-style output
        return (

            spk3_rec,

            mem3_rec
        )


# ============================================================
# Initialize model
# ============================================================

model = SpikingMNISTModel().to(
    device
)


print("\nModel:")
print(model)


# ============================================================
# Number of trainable parameters
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
#
# Keep SGD exactly as the original LIF baseline.
# ============================================================

optimizer = optim.SGD(

    model.parameters(),

    lr=learning_rate
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(

    model,

    device,

    train_loader,

    optimizer,

    criterion,

    epoch
):


    model.train()


    # ========================================================
    # Statistics
    # ========================================================

    running_loss = 0.0

    correct = 0

    total = 0


    # Layer firing-rate accumulators
    fr1_total = 0.0

    fr2_total = 0.0

    fr3_total = 0.0


    num_batches = 0


    # ========================================================
    # Training batches
    # ========================================================

    for batch_idx, (
        data,
        target
    ) in enumerate(
        train_loader
    ):


        data = data.to(
            device
        )


        target = target.to(
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

            mem3

        ) = model(

            data,

            return_all=True
        )


        # ====================================================
        # Temporal output
        #
        # Same method as your original LIF implementation:
        #
        # Sum output membrane potential over time.
        #
        # [T, B, 10]
        #
        #      ↓
        #
        # [B, 10]
        # ====================================================

        output = mem3.sum(
            dim=0
        )


        # ====================================================
        # Loss
        # ====================================================

        loss = criterion(

            output,

            target
        )


        # ====================================================
        # Backpropagation Through Time
        # ====================================================

        loss.backward()


        optimizer.step()


        # ====================================================
        # Loss statistics
        # ====================================================

        running_loss += loss.item()


        # ====================================================
        # Prediction
        # ====================================================

        predicted = output.argmax(
            dim=1
        )


        total += target.size(
            0
        )


        correct += (

            predicted
            ==
            target

        ).sum().item()


        # ====================================================
        # Firing rates
        #
        # Since spikes are binary:
        #
        # mean(spikes)
        #
        # = fraction of neuron-time events that fired.
        # ====================================================

        fr1 = spk1.detach().mean().item()

        fr2 = spk2.detach().mean().item()

        fr3 = spk3.detach().mean().item()


        fr1_total += fr1

        fr2_total += fr2

        fr3_total += fr3


        num_batches += 1


        # ====================================================
        # Intermediate output
        #
        # Same style as MTJ version
        # ====================================================

        if batch_idx % 100 == 0:

            print(

                f"Epoch {epoch} | "

                f"Batch {batch_idx:4d}/{len(train_loader)} | "

                f"Loss {loss.item():.4f} | "

                f"Spike FR: "

                f"{fr1:.3f}, "

                f"{fr2:.3f}, "

                f"{fr3:.3f}"
            )


    # ========================================================
    # Epoch statistics
    # ========================================================

    epoch_loss = (

        running_loss

        /

        len(train_loader)
    )


    epoch_accuracy = (

        100.0

        *

        correct

        /

        total
    )


    # ========================================================
    # Average firing rates
    # ========================================================

    fr1_avg = (

        fr1_total

        /

        num_batches
    )


    fr2_avg = (

        fr2_total

        /

        num_batches
    )


    fr3_avg = (

        fr3_total

        /

        num_batches
    )


    return {

        "loss": epoch_loss,

        "accuracy": epoch_accuracy,

        "fr1": fr1_avg,

        "fr2": fr2_avg,

        "fr3": fr3_avg
    }


# ============================================================
# Test function
# ============================================================

def evaluate(

    model,

    device,

    test_loader
):


    model.eval()


    correct = 0

    total = 0


    # ========================================================
    # Firing-rate accumulators
    # ========================================================

    fr1_total = 0.0

    fr2_total = 0.0

    fr3_total = 0.0


    num_batches = 0


    with torch.no_grad():


        for data, target in test_loader:


            data = data.to(
                device
            )


            target = target.to(
                device
            )


            # =================================================
            # Forward
            # =================================================

            (

                spk1,

                spk2,

                spk3,

                mem3

            ) = model(

                data,

                return_all=True
            )


            # =================================================
            # Sum output membrane potentials over time
            # =================================================

            output = mem3.sum(
                dim=0
            )


            # =================================================
            # Classification
            # =================================================

            predicted = output.argmax(
                dim=1
            )


            total += target.size(
                0
            )


            correct += (

                predicted
                ==
                target

            ).sum().item()


            # =================================================
            # Firing rates
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


            num_batches += 1


    # ========================================================
    # Test accuracy
    # ========================================================

    accuracy = (

        100.0

        *

        correct

        /

        total
    )


    # ========================================================
    # Average firing rates
    # ========================================================

    fr1_avg = (

        fr1_total

        /

        num_batches
    )


    fr2_avg = (

        fr2_total

        /

        num_batches
    )


    fr3_avg = (

        fr3_total

        /

        num_batches
    )


    return {

        "accuracy": accuracy,

        "fr1": fr1_avg,

        "fr2": fr2_avg,

        "fr3": fr3_avg
    }


# ============================================================
# Training
# ============================================================

best_accuracy = 0.0


for epoch in range(
    1,
    num_epochs + 1
):


    print(
        "\n===================================================="
    )


    print(
        f"Epoch {epoch}/{num_epochs}"
    )


    print(
        "===================================================="
    )


    # ========================================================
    # Train
    # ========================================================

    train_stats = train_one_epoch(

        model,

        device,

        train_loader,

        optimizer,

        criterion,

        epoch
    )


    # ========================================================
    # Test
    # ========================================================

    test_stats = evaluate(

        model,

        device,

        test_loader
    )


    # ========================================================
    # Print training results
    # ========================================================

    print(
        "\n---------------- TRAIN ----------------"
    )


    print(
        f"Loss:           "
        f"{train_stats['loss']:.4f}"
    )


    print(
        f"Accuracy:       "
        f"{train_stats['accuracy']:.2f}%"
    )


    print(
        "\nOutput spike firing rates:"
    )


    print(
        f"Layer 1:        "
        f"{train_stats['fr1']:.4f}"
    )


    print(
        f"Layer 2:        "
        f"{train_stats['fr2']:.4f}"
    )


    print(
        f"Layer 3:        "
        f"{train_stats['fr3']:.4f}"
    )


    # ========================================================
    # Print test results
    # ========================================================

    print(
        "\n---------------- TEST ----------------"
    )


    print(
        f"Accuracy:       "
        f"{test_stats['accuracy']:.2f}%"
    )


    print(
        "\nOutput spike firing rates:"
    )


    print(
        f"Layer 1:        "
        f"{test_stats['fr1']:.4f}"
    )


    print(
        f"Layer 2:        "
        f"{test_stats['fr2']:.4f}"
    )


    print(
        f"Layer 3:        "
        f"{test_stats['fr3']:.4f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if test_stats["accuracy"] > best_accuracy:


        best_accuracy = test_stats[
            "accuracy"
        ]


        torch.save(

            model.state_dict(),

            "best_mnist_lif_mlp.pth"
        )


        print(
            "\nBest model saved."
        )


# ============================================================
# Training complete
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

    f"Best Test Accuracy: "

    f"{best_accuracy:.2f}%"
)


# ============================================================
# Load best model
# ============================================================

model.load_state_dict(

    torch.load(

        "best_mnist_lif_mlp.pth",

        map_location=device
    )
)


# ============================================================
# Final test
# ============================================================

final_stats = evaluate(

    model,

    device,

    test_loader
)


print(

    f"\nFinal Test Accuracy: "

    f"{final_stats['accuracy']:.2f}%"
)


print(
    "\nFinal output firing rates:"
)


print(

    f"Layer 1 = "

    f"{final_stats['fr1']:.4f}"
)


print(

    f"Layer 2 = "

    f"{final_stats['fr2']:.4f}"
)


print(

    f"Layer 3 = "

    f"{final_stats['fr3']:.4f}"
)


# ============================================================
# Final evaluation using spike counts
# ============================================================

def test_spike_count(

    model,

    device,

    test_loader
):


    model.eval()


    correct = 0

    total = 0


    with torch.no_grad():


        for data, target in test_loader:


            data = data.to(
                device
            )


            target = target.to(
                device
            )


            # =================================================
            # Forward
            # =================================================

            spk_rec, _ = model(
                data
            )


            # =================================================
            # Count output spikes
            #
            # [T, B, 10]
            #
            #      ↓
            #
            # [B, 10]
            # =================================================

            spike_count = spk_rec.sum(
                dim=0
            )


            predicted = spike_count.argmax(
                dim=1
            )


            total += target.size(
                0
            )


            correct += (

                predicted
                ==
                target

            ).sum().item()


    accuracy = (

        100.0

        *

        correct

        /

        total
    )


    print(

        f"\nSpike-count Test Accuracy: "

        f"{accuracy:.2f}%"
    )


    return accuracy


# ============================================================
# Spike-count test
# ============================================================

test_spike_count(

    model,

    device,

    test_loader
)


# ============================================================
# Save final model
# ============================================================

torch.save(

    model.state_dict(),

    "mnist_snn_mlp.pth"
)


print(

    "\nModel saved as mnist_snn_mlp.pth"
)