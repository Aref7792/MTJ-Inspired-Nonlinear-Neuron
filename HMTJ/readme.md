# MNIST SNN with Hard MTJ Pulse/No-Pulse Dynamics

This project implements a spiking neural network (SNN) for MNIST classification using a custom **MTJ-inspired nonlinear neuron** with a **hard binary pulse/no-pulse decision in the forward pass**.

The network architecture is:

```text
784 -> 128 -> 64 -> 10
```

The main idea is:

```text
z > 0   -> pulse = 1 -> MTJ integration
z <= 0  -> pulse = 0 -> MTJ leakage
```

During backpropagation, a sigmoid-based straight-through estimator (STE) is used so that gradients can still propagate through the hard pulse decision.

---

## Architecture

```text
MNIST image
   |
   v
Flatten: 28x28 -> 784
   |
   v
Linear(784, 128)
   |
   v
Hard MTJ Neuron
   |
   v
Linear(128, 64)
   |
   v
Hard MTJ Neuron
   |
   v
Linear(64, 10)
   |
   v
Hard MTJ Neuron
   |
   v
Output Spike Counts
   |
   v
Predicted Digit
```

The network is simulated for:

```python
NUM_STEPS = 25
```

time steps.

---

## Hard Pulse Decision

The forward pulse is binary:

\[
p(z)=
\begin{cases}
1, & z>0,\\
0, & z\le 0.
\end{cases}
\]

Therefore:

- `pulse = 1` means the neuron follows the MTJ integration dynamics.
- `pulse = 0` means the neuron follows the MTJ leakage dynamics.

The forward behavior is therefore physically hard rather than a soft interpolation.

---

## Straight-Through Estimator

A hard threshold is not differentiable, so the implementation uses a straight-through estimator.

The soft surrogate is

\[
s(z)=\sigma(\alpha z)
=
\frac{1}{1+\exp(-\alpha z)}.
\]

The hard value is

\[
h(z)=\mathbb{1}[z>0].
\]

The implemented pulse is

\[
p
=
h+s-\operatorname{detach}(s).
\]

In PyTorch:

```python
soft = torch.sigmoid(alpha * z)
hard = (z > 0).to(z.dtype)

pulse = hard + soft - soft.detach()
```

### Forward Pass

Because

```text
soft - soft.detach() = 0
```

numerically in the forward pass,

\[
p=h.
\]

So the pulse is exactly binary.

### Backward Pass

During backpropagation, the gradient flows through the sigmoid term.

The hyperparameter controlling the surrogate steepness is:

```python
PULSE_SURROGATE_ALPHA = 5.0
```

Larger values make the surrogate gradient more concentrated near \(z=0\).

---

## Positive Synaptic Input to Current Density

Pulse existence and pulse strength are handled separately.

The pulse exists only if:

\[
z>0.
\]

For pulse strength, the positive part of \(z\) is used:

\[
z_+ = \max(z,0).
\]

The normalized current strength is

\[
r(z)
=
1-\exp(-z_+).
\]

This gives:

\[
0\le r(z)<1.
\]

The physical current density is then

\[
J
=
J_{\min}
+
r(z)
\left(
J_{\max}-J_{\min}
\right).
\]

The code uses:

```python
J_MIN = 1e11
J_MAX = 1e12
```

so

\[
10^{11}
\le J \le
10^{12}
\quad \mathrm{A/m^2}.
\]

The `z -> J` conversion is a neural-to-device interface mapping used by the implementation.

---

## MTJ Current-Dependent Rise Time

The rise-time constant is modeled as:

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

The units are picoseconds.

As the current density increases, \(\tau_r\) decreases, producing faster magnetization evolution.

---

## MTJ Integration Dynamics

When a pulse exists, the neuron follows the MTJ integration equation:

\[
m_{\mathrm{int}}
=
A(J)
\tanh
\left[
\frac{T_p}{\tau_r(J)}
+
\tanh^{-1}
\left(
\frac{m}{A(J)}
\right)
\right].
\]

The pulse width is fixed:

```python
PULSE_WIDTH_PS = 30.0
```

Unlike the earlier soft-drive implementation, the pulse width is not continuously multiplied by a drive value in this hard-pulse model.

---

## MTJ Leakage Dynamics

When no pulse exists, the neuron follows the nonlinear leakage equation.

First,

\[
B
=
\exp
\left(
-\frac{\Delta t}{\tau_l}
\right).
\]

Then,

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

The default timing parameters are:

```python
DT_PS = 100.0
TAU_LEAK_PS = 503.8
```

---

## Hard Integrate/Leak Selection

The state update is:

\[
m_{\mathrm{new}}
=
p\,m_{\mathrm{int}}
+
(1-p)m_{\mathrm{leak}}.
\]

Because \(p\in\{0,1\}\) in the forward pass, this is not a fractional mixture.

If

\[
p=1,
\]

then

\[
m_{\mathrm{new}}=m_{\mathrm{int}}.
\]

If

\[
p=0,
\]

then

\[
m_{\mathrm{new}}=m_{\mathrm{leak}}.
\]

So the model selects exactly one physical branch per neuron per time step.

---

## Output Spike Generation

The magnetization state \(m_z\) acts as the neuron state.

The firing threshold is:

```python
THRESHOLD = 0.8
```

An output spike is generated from:

\[
m_z-\theta,
\]

using the snnTorch fast-sigmoid surrogate:

```python
output_spike_grad = surrogate.fast_sigmoid(
    slope=25
)
```

This surrogate is used for training through the binary output-spike operation.

---

## No Hard Reset

The implementation does not reset the magnetization state to zero after a spike.

The state is preserved and continues evolving according to either the integration or leakage equation at the next time step.

---

## Classification

The output layer contains 10 spiking neurons corresponding to the 10 MNIST classes.

The spike count for class \(c\) is:

\[
S_c
=
\sum_{t=1}^{T}
s_c[t].
\]

The predicted class is:

\[
\hat{y}
=
\arg\max_c S_c.
\]

Training also uses the accumulated output spike counts:

```python
output = spk3.sum(dim=0)
loss = criterion(output, labels)
```

with cross-entropy loss.

---

## Training Configuration

The default training settings are:

```python
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_STEPS = 25
GRAD_CLIP = 1.0
```

The optimizer is Adam:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
```

Gradient clipping is applied during BPTT:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    GRAD_CLIP
)
```

---

## Dataset

The implementation uses MNIST from `torchvision.datasets`.

Preprocessing is:

```python
transform = transforms.ToTensor()
```

so the input pixels remain in:

\[
[0,1].
\]

---

## Firing-Rate Monitoring

The code reports output-spike firing rates for all three MTJ layers:

```text
Layer 1
Layer 2
Layer 3
```

The firing rate is computed from the mean of the binary spike tensor.

This is useful for evaluating activity sparsity and comparing the hard-MTJ model with conventional LIF or soft-MTJ models.

---

## Hard Pulse-Rate Monitoring

In addition to output-spike firing rate, this implementation explicitly reports the **hard pulse rate** of each layer.

The pulse rate measures the fraction of neuron-time events for which:

\[
z>0.
\]

The reported quantities are:

```text
Hard pulse rates:
Layer 1
Layer 2
Layer 3
```

This is different from the output firing rate.

A neuron may receive a pulse without necessarily reaching its output firing threshold.

Therefore:

```text
pulse event != output spike
```

in general.

---

## Installation

Create a Conda environment:

```bash
conda create -n mnist_snn python=3.10 -y
conda activate mnist_snn
```

Install the required packages:

```bash
pip install torch torchvision snntorch
```

Optional notebook support:

```bash
pip install jupyter ipykernel matplotlib
```

---

## Running

Save the Python script as, for example:

```text
mnist_hard_mtj_snn.py
```

Run:

```bash
python mnist_hard_mtj_snn.py
```

MNIST is downloaded automatically if needed.

---

## Output

During training, the script reports statistics such as:

```text
Epoch 1 | Batch    0/938 | Loss 2.3026 | Spike FR: 0.100, 0.050, 0.020 | Pulse Rate: 0.500, 0.450, 0.400
```

After each epoch it prints:

```text
---------------- TRAIN ----------------

Loss:
Accuracy:

Output spike firing rates:
Layer 1:
Layer 2:
Layer 3:

Hard pulse rates:
Layer 1:
Layer 2:
Layer 3:

---------------- TEST ----------------

Accuracy:

Output spike firing rates:
Layer 1:
Layer 2:
Layer 3:

Hard pulse rates:
Layer 1:
Layer 2:
Layer 3:
```

The best checkpoint is saved as:

```text
best_hard_mtjlif_mnist.pth
```

---

## Main Hyperparameters

| Parameter | Default | Meaning |
|---|---:|---|
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `NUM_EPOCHS` | 10 | Number of epochs |
| `NUM_STEPS` | 25 | SNN simulation steps |
| `THRESHOLD` | 0.8 | MTJ output firing threshold |
| `PULSE_WIDTH_PS` | 30 ps | Fixed integration pulse width |
| `DT_PS` | 100 ps | Leak interval |
| `TAU_LEAK_PS` | 503.8 ps | Leakage time constant |
| `J_MIN` | \(10^{11}\) A/m² | Minimum current density |
| `J_MAX` | \(10^{12}\) A/m² | Maximum current density |
| `PULSE_SURROGATE_ALPHA` | 5.0 | STE sigmoid steepness |
| `GRAD_CLIP` | 1.0 | Maximum gradient norm |

---

## Hard vs. Soft MTJ Neuron

The important distinction between this implementation and a soft integrate/leak formulation is:

### Soft formulation

\[
m_{\mathrm{new}}
=
d\,m_{\mathrm{int}}
+
(1-d)m_{\mathrm{leak}},
\qquad
0<d<1.
\]

The neuron can partially integrate and partially leak at the same time.

### Hard formulation

\[
m_{\mathrm{new}}
=
p\,m_{\mathrm{int}}
+
(1-p)m_{\mathrm{leak}},
\qquad
p\in\{0,1\}.
\]

The neuron selects exactly one branch in the forward pass.

The sigmoid is used only to approximate the gradient of this hard pulse gate during optimization.

---

## Summary

The neuron can be summarized as:

\[
z_t
\rightarrow
p_t=H(z_t)
\]

with the STE used only during backpropagation.

For a positive pulse:

\[
z_t>0
\Rightarrow
J_t=f(z_t)
\Rightarrow
m_{t+1}
=
f_{\mathrm{integrate}}
(m_t,J_t).
\]

For no pulse:

\[
z_t\le0
\Rightarrow
m_{t+1}
=
f_{\mathrm{leak}}
(m_t).
\]

Finally:

\[
s_t
=
H(m_t-\theta).
\]

This provides a hard event-driven MTJ-inspired neuron model while retaining trainability through surrogate gradients.
