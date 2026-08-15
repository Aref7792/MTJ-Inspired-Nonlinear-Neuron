# MNIST-10 Classification with LIF and MTJ-Inspired Spiking Neurons

This repository compares three spiking-neuron models for **MNIST 10-class handwritten digit classification** using the same network architecture:

```text
784 -> 128 -> 64 -> 10
```

Each `28 x 28` MNIST image is flattened to 784 features and processed by three fully connected spiking layers.

## Repository Structure

```text
BLIF/
SMTJ/
HMTJ/
```

- **BLIF** — standard binary Leaky Integrate-and-Fire (LIF) baseline implemented with `snnTorch`.
- **SMTJ** — MTJ-inspired magnetic neuron with **soft pulse modeling** and smooth integrate/leak behavior.
- **HMTJ** — MTJ-inspired magnetic neuron with **hard pulse/no-pulse modeling**. In the forward pass, positive input selects integration and non-positive input selects leakage; a straight-through estimator is used for training.

All three implementations use the same `784 -> 128 -> 64 -> 10` architecture so that the effect of the neuron model can be compared directly.

## Model Summary

| Folder | Neuron Model | Main Behavior |
|---|---|---|
| `BLIF` | Standard LIF | Conventional leaky integrate-and-fire dynamics |
| `SMTJ` | Soft MTJ | Smooth magnetic integration/leakage with soft pulse modeling |
| `HMTJ` | Hard MTJ | Binary pulse/no-pulse selection between integration and leakage |

## Installation

Create and activate a Python environment:

```bash
conda create -n mnist_snn python=3.10 -y
conda activate mnist_snn
```

Install the required packages:

```bash
pip install torch torchvision snntorch numpy
```

The MNIST dataset is downloaded automatically through `torchvision` when the training scripts are run.

Run each implementation from its corresponding folder, for example:

```bash
cd BLIF
python mnist_lif_mlp.py
```

```bash
cd SMTJ
python mnist_mtj_soft.py
```

```bash
cd HMTJ
python mnist_mtj_hard.py
```

## Results

### Classification Accuracy

| Model | Best Test Accuracy | Final Test Accuracy | Spike-Count Test Accuracy |
|---|---:|---:|---:|
| **BLIF** | 96.93% | 96.93% | **97.12%** |
| **SMTJ** | **96.99%** | **96.99%** | — |
| **HMTJ** | 95.25% | 95.25% | — |

For the main final-test metric, SMTJ achieves the highest accuracy at **96.99%**, slightly above BLIF at **96.93%**. BLIF reaches **97.12%** when classification is performed using output spike counts. HMTJ obtains **95.25%**.

### Final Output Firing Rates

| Model | Layer 1 | Layer 2 | Layer 3 |
|---|---:|---:|---:|
| **BLIF** | 0.0874 | 0.0492 | 0.0872 |
| **SMTJ** | 0.2233 | 0.4341 | 0.1614 |
| **HMTJ** | 0.3475 | 0.3402 | 0.1956 |

BLIF has the lowest firing activity in all three layers. SMTJ achieves nearly the same classification accuracy as BLIF but with substantially higher activity, especially in Layer 2. HMTJ shows the highest Layer 1 and Layer 3 firing rates while also giving the lowest classification accuracy of the three models.

### HMTJ Hard Pulse Rates

| Layer | Hard Pulse Rate |
|---|---:|
| Layer 1 | 0.3713 |
| Layer 2 | 0.3885 |
| Layer 3 | 0.3050 |

The HMTJ pulse rates are higher than its output firing rates because a pulse event causes the neuron to follow the integration branch but does not necessarily make the neuron cross its output firing threshold.

## Overall Comparison

The experiments show that changing the neuron dynamics while keeping the network architecture fixed produces measurable differences in both accuracy and activity.

- **BLIF** provides the strongest baseline in terms of low firing activity and reaches **97.12%** with spike-count decoding.
- **SMTJ** gives the highest standard final test accuracy, **96.99%**, while using a differentiable soft magnetic-neuron formulation.
- **HMTJ** provides the most hardware-like pulse/no-pulse behavior of the MTJ variants, but its final accuracy is lower at **95.25%**.

The repository therefore provides a controlled comparison between conventional LIF neurons and soft/hard MTJ-inspired magnetic neuron dynamics on the same MNIST classification problem.
