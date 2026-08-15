# MNIST-10 Classification with Spiking and MTJ-Inspired Neuron Models

This repository is used for **MNIST 10-class handwritten digit classification**.

The goal is to compare a conventional binary spiking neuron baseline with MTJ-inspired magnetic neuron models under the same multilayer perceptron architecture.

The MNIST task contains 10 output classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Each input image is a grayscale handwritten digit of size:

```text
28 x 28
```

The image is flattened into a 784-dimensional vector before being passed to the network.

---

## Network Architecture

All implementations use the same MLP structure:

```text
784 -> 128 -> 64 -> 10
```

The overall processing pipeline is:

```text
MNIST Image
    |
    v
28 x 28
    |
    v
Flatten
    |
    v
784
    |
    v
Fully Connected Layer
784 -> 128
    |
    v
Spiking / MTJ Neuron Layer
    |
    v
Fully Connected Layer
128 -> 64
    |
    v
Spiking / MTJ Neuron Layer
    |
    v
Fully Connected Layer
64 -> 10
    |
    v
Output Spiking / MTJ Neuron Layer
    |
    v
Digit Classification
```

Keeping the architecture fixed makes it possible to compare the effect of the neuron model itself.

---

# Repository Structure

The repository contains three main implementations:

```text
BLIF/
SMTJ/
HMTJ/
```

Each folder uses the same MNIST classification architecture but a different neuron model.

---

## BLIF — Binary LIF Baseline

The `BLIF` folder contains the baseline implementation based on standard **Leaky Integrate-and-Fire (LIF)** neurons.

The architecture is:

```text
784 -> FC -> LIF
    -> FC -> LIF
    -> FC -> LIF
```

The LIF neurons are implemented using `snnTorch`.

A typical membrane update follows the idea

\[
U[t+1]
=
\beta U[t]
+
I[t+1],
\]

together with threshold-based spike generation.

The main LIF parameter is:

```python
beta = 0.9
```

which controls the membrane decay.

Because spike generation is non-differentiable, a surrogate-gradient function is used during training.

The BLIF model serves as the conventional SNN baseline for comparison with the magnetic-neuron implementations.

### BLIF Purpose

`BLIF` is used to measure:

- classification accuracy,
- output spike activity,
- layer firing rates,
- performance of a conventional LIF-based SNN.

---

## SMTJ — Soft-Pulse Magnetic Neuron

The `SMTJ` folder contains the **soft MTJ-inspired magnetic neuron model**.

Instead of a standard LIF membrane equation, the neuron state is modeled using nonlinear magnetic dynamics.

The neuron state can be interpreted as the MTJ magnetization variable:

\[
m_z.
\]

The SMTJ model uses a continuous, differentiable pulse representation.

The neural input is first mapped to a normalized drive using a sigmoid:

\[
d(z)
=
\sigma(z)
=
\frac{1}{1+e^{-z}}.
\]

The normalized drive is then used to determine the physical excitation applied to the MTJ-inspired neuron.

For example, the current density is mapped as

\[
J
=
J_{\min}
+
d(z)
\left(
J_{\max}-J_{\min}
\right).
\]

The drive is also used to determine the effective pulse width in the soft-pulse model:

\[
T_{\mathrm{eff}}
=
d(z)T_{\mathrm{pulse}}.
\]

---

### SMTJ Integration Dynamics

During integration, the magnetic state follows

\[
m_{\mathrm{int}}
=
A(J)
\tanh
\left[
\frac{T_{\mathrm{eff}}}{\tau_r(J)}
+
\tanh^{-1}
\left(
\frac{m}{A(J)}
\right)
\right].
\]

The rise-time constant depends on current density:

\[
\tau_r(J)
=
386.98
\left(
\frac{J}{10^{11}}
\right)^{-1.223}
+
8.88.
\]

---

### SMTJ Leakage Dynamics

When the neuron leaks, the state follows

\[
B
=
\exp
\left(
-\frac{\Delta t}{\tau_l}
\right),
\]

and

\[
m_{\mathrm{leak}}
=
\frac{
mB
}{
\sqrt{
1-m^2(1-B^2)
}
}.
\]

---

### Soft Integrate/Leak Selection

In the soft model, integration and leakage can be smoothly combined.

A soft gate \(g\) can be used:

\[
m_{\mathrm{new}}
=
g\,m_{\mathrm{int}}
+
(1-g)m_{\mathrm{leak}}.
\]

When \(g\) is close to 1, the neuron behaves mostly as an integrating neuron.

When \(g\) is close to 0, the neuron behaves mostly as a leaking neuron.

Because the transition is continuous, the SMTJ model is fully differentiable and is easier to train with gradient-based optimization.

### SMTJ Purpose

`SMTJ` is used to study:

- differentiable MTJ-inspired neuron dynamics,
- soft pulse modeling,
- nonlinear magnetic integration,
- nonlinear magnetic leakage,
- firing-rate behavior,
- classification performance.

---

## HMTJ — Hard-Pulse Magnetic Neuron

The `HMTJ` folder contains the **hard pulse/no-pulse MTJ-inspired neuron model**.

The main difference from SMTJ is that the forward pulse decision is binary.

The pulse rule is:

\[
p(z)
=
\begin{cases}
1, & z>0,\\
0, & z\le0.
\end{cases}
\]

Therefore:

```text
z > 0
    -> pulse exists
    -> MTJ integration

z <= 0
    -> no pulse
    -> MTJ leakage
```

The forward pass therefore selects exactly one physical branch.

---

### Hard MTJ State Update

The state update is

\[
m_{\mathrm{new}}
=
p\,m_{\mathrm{int}}
+
(1-p)m_{\mathrm{leak}}.
\]

Since

\[
p\in\{0,1\},
\]

the neuron does not partially integrate and leak at the same time.

If

\[
p=1,
\]

then

\[
m_{\mathrm{new}}
=
m_{\mathrm{int}}.
\]

If

\[
p=0,
\]

then

\[
m_{\mathrm{new}}
=
m_{\mathrm{leak}}.
\]

---

### Straight-Through Gradient

The hard pulse decision is non-differentiable.

To train the network, HMTJ uses a sigmoid-based straight-through estimator.

The soft surrogate is

\[
s(z)
=
\sigma(\alpha z).
\]

The hard decision is

\[
h(z)
=
\mathbb{1}[z>0].
\]

The implemented pulse is

\[
p
=
h+s-\operatorname{detach}(s).
\]

This gives:

- hard binary behavior in the forward pass,
- differentiable surrogate behavior in the backward pass.

Therefore, the HMTJ model is closer to an event-driven pulse/no-pulse implementation while remaining trainable.

### HMTJ Purpose

`HMTJ` is used to study:

- hard pulse/no-pulse behavior,
- binary integrate/leak selection,
- straight-through gradient training,
- MTJ-inspired nonlinear dynamics,
- pulse rates,
- firing rates,
- classification performance.

---

# Comparison of the Three Models

| Folder | Neuron Model | Integrate/Leak Behavior | Training Method |
|---|---|---|---|
| `BLIF` | Standard LIF | LIF membrane dynamics | Surrogate gradient |
| `SMTJ` | Soft MTJ-inspired neuron | Smooth integration/leakage | Fully differentiable soft modeling |
| `HMTJ` | Hard MTJ-inspired neuron | Binary integrate or leak | Straight-through estimator |

The main experimental objective is to compare these neuron models while keeping the network architecture fixed.

---

# Main Experimental Quantities

The implementations can be compared using:

- MNIST classification accuracy,
- output spike-count accuracy,
- firing rate of each layer,
- pulse rate for MTJ-based models,
- training stability,
- sensitivity to temporal simulation length,
- sensitivity to MTJ pulse parameters.

Because all models use

```text
784 -> 128 -> 64 -> 10
```

the comparison focuses primarily on the effect of neuron dynamics rather than network size.

---

# MNIST Dataset

MNIST contains grayscale handwritten digit images.

Each image has shape:

```text
1 x 28 x 28
```

The image is converted to a tensor and flattened:

\[
28\times28
=
784.
\]

The network output contains 10 neurons corresponding to:

```text
0
1
2
3
4
5
6
7
8
9
```

The predicted digit is determined from the output activity of these 10 neurons.

---

# Temporal Processing

All three SNN implementations operate over multiple simulation time steps.

A typical configuration is:

```python
NUM_STEPS = 25
```

or

```python
num_steps = 25
```

depending on the implementation.

The network therefore processes each input over a temporal window rather than using only a single instantaneous forward pass.

---

# Firing Rate

For a binary spike tensor, the firing rate is calculated as

\[
FR
=
\frac{
\text{total number of spikes}
}{
\text{number of neuron-time events}
}.
\]

In practice this corresponds to the mean of the binary spike tensor.

Firing-rate measurements are reported for:

```text
Layer 1
Layer 2
Layer 3
```

and provide an activity-based comparison between BLIF, SMTJ, and HMTJ.

---

# Overall Goal

The repository provides a controlled comparison between:

\[
\boxed{\text{Standard LIF}}
\]

and

\[
\boxed{\text{MTJ-inspired magnetic neurons}}
\]

for the same MNIST 10-class classification problem.

The three implementations represent progressively different neuron dynamics:

```text
BLIF
|
| Standard binary LIF neuron
|
v

SMTJ
|
| MTJ-inspired neuron
| with soft pulse / soft integrate-leak behavior
|
v

HMTJ
|
| MTJ-inspired neuron
| with hard pulse/no-pulse behavior
|
v
```

This structure allows the effect of magnetic-neuron dynamics and pulse modeling to be evaluated independently of the network architecture.


# Installation and Setup

## 1. Clone the Repository

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace `<YOUR_REPOSITORY_URL>` and `<YOUR_REPOSITORY_NAME>` with the actual GitHub repository information.

---

## 2. Create a Python Environment

Using Conda is recommended:

```bash
conda create -n mnist_snn python=3.10 -y
conda activate mnist_snn
```

A standard Python virtual environment can also be used:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install Required Packages

The implementations require:

- Python
- PyTorch
- torchvision
- snnTorch

Install them with:

```bash
pip install torch torchvision snntorch
```

For notebook-based experimentation, visualization, or plotting, the following packages are also useful:

```bash
pip install jupyter ipykernel matplotlib numpy
```

A complete installation command is:

```bash
pip install torch torchvision snntorch numpy matplotlib jupyter ipykernel
```

---

# Suggested Requirements File

A minimal `requirements.txt` can contain:

```text
torch
torchvision
snntorch
numpy
matplotlib
```

Install all dependencies from the requirements file using:

```bash
pip install -r requirements.txt
```

---

# Recommended Repository Organization

A suggested structure is:

```text
MNIST-MTJ-SNN/
|
|-- README.md
|-- requirements.txt
|
|-- BLIF/
|   |-- README.md
|   |-- mnist_lif_mlp.py
|
|-- SMTJ/
|   |-- README.md
|   |-- mnist_mtj_soft.py
|
|-- HMTJ/
|   |-- README.md
|   |-- mnist_mtj_hard.py
|
|-- data/
    |-- MNIST/
```

The exact Python filenames can be changed, but the three main folders should correspond to the three neuron models described above.

---

# Running the Models

Each implementation can be run independently.

## BLIF

Move to the BLIF folder:

```bash
cd BLIF
```

Run the standard LIF baseline:

```bash
python mnist_lif_mlp.py
```

This implementation uses standard `snnTorch` LIF neurons.

---

## SMTJ

Move to the SMTJ folder:

```bash
cd SMTJ
```

Run the soft-pulse magnetic-neuron model:

```bash
python mnist_mtj_soft.py
```

This implementation uses differentiable MTJ-inspired integration and leakage with soft pulse modeling.

---

## HMTJ

Move to the HMTJ folder:

```bash
cd HMTJ
```

Run the hard-pulse magnetic-neuron model:

```bash
python mnist_mtj_hard.py
```

This implementation uses binary pulse/no-pulse behavior in the forward pass and a straight-through estimator for gradient propagation.

---

# Dataset Download

The MNIST dataset is loaded through:

```python
torchvision.datasets.MNIST
```

with:

```python
download=True
```

Therefore, the dataset is downloaded automatically the first time one of the training scripts is executed.

The input preprocessing uses:

```python
transforms.ToTensor()
```

so pixel values remain in:

\[
[0,1].
\]

No additional image normalization is required by the current implementations.

---

# GPU Support

The scripts automatically use available hardware acceleration when supported.

Typical device selection includes:

```python
torch.device("cuda")
```

for NVIDIA GPUs,

```python
torch.device("mps")
```

for Apple Silicon GPUs, and

```python
torch.device("cpu")
```

otherwise.

For NVIDIA GPUs, make sure that the installed PyTorch version is compatible with the system CUDA configuration.

You can verify CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

To verify `snnTorch`:

```bash
python -c "import snntorch; print('snnTorch installed successfully')"
```

---

# Training

The current implementations use MNIST training and test splits directly from `torchvision`.

Typical training settings are:

```text
Batch size:        64
Epochs:            10
Simulation steps:  25
```

Optimizer settings differ between implementations according to the corresponding experiment.

For example:

```text
BLIF:
    SGD

SMTJ:
    Adam

HMTJ:
    Adam
```

When comparing the neuron models experimentally, optimizer and training differences should be reported clearly.

---

# Model Outputs

Depending on the implementation, the scripts report quantities such as:

```text
Training loss
Training accuracy
Test accuracy
Layer 1 firing rate
Layer 2 firing rate
Layer 3 firing rate
```

The hard MTJ implementation additionally reports:

```text
Layer 1 hard pulse rate
Layer 2 hard pulse rate
Layer 3 hard pulse rate
```

The BLIF implementation also includes a final output spike-count classification evaluation.

---

# Saved Checkpoints

The scripts save the best-performing trained models during training.

Typical checkpoint names include:

```text
BLIF:
    best_mnist_lif_mlp.pth
    mnist_snn_mlp.pth

SMTJ:
    best_mtjlif_mnist.pth

HMTJ:
    best_hard_mtjlif_mnist.pth
```

These files contain the PyTorch model state dictionaries and can be reloaded using:

```python
model.load_state_dict(
    torch.load(
        "checkpoint_name.pth",
        map_location=device
    )
)
```

---

# Reproducibility

The implementations use a fixed PyTorch random seed:

```python
torch.manual_seed(42)
```

and, when CUDA is available:

```python
torch.cuda.manual_seed_all(42)
```

This helps reduce run-to-run variation.

For stricter deterministic CUDA behavior, additional PyTorch deterministic settings may be enabled if required.

---

# Experimental Comparison

For a fair comparison between BLIF, SMTJ, and HMTJ, it is recommended to report at least:

```text
Best test accuracy
Final test accuracy
Layer 1 firing rate
Layer 2 firing rate
Layer 3 firing rate
Number of simulation steps
Number of trainable parameters
Training optimizer
Learning rate
```

For HMTJ, also report:

```text
Layer 1 hard pulse rate
Layer 2 hard pulse rate
Layer 3 hard pulse rate
```

For SMTJ experiments involving the integrate/leak parametric sigmoid, also report:

```text
Gate sigmoid slope
Gate sigmoid threshold
```

This makes the comparison between smooth and hard switching behavior transparent.

---

# Quick Start

After cloning the repository:

```bash
conda create -n mnist_snn python=3.10 -y
conda activate mnist_snn

pip install torch torchvision snntorch numpy matplotlib
```

Then run one of the models:

```bash
cd BLIF
python mnist_lif_mlp.py
```

or:

```bash
cd SMTJ
python mnist_mtj_soft.py
```

or:

```bash
cd HMTJ
python mnist_mtj_hard.py
```

---

# Notes

The three folders represent different neuron-dynamics assumptions rather than different network architectures.

The shared architecture is always:

```text
784 -> 128 -> 64 -> 10
```

Therefore, the repository is designed to study how the neuron model and pulse representation affect MNIST classification behavior while maintaining a comparable network structure.
