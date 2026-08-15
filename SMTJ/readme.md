# MNIST SNN with MTJ-Inspired Nonlinear Neuron and Parametric Integrate/Leak Gate

This project implements a spiking neural network (SNN) for MNIST classification using a custom **MTJ-inspired nonlinear neuron model**.

The fully connected architecture is:

```text
784 -> 128 -> 64 -> 10
```

The implementation preserves the original smooth drive calculation and introduces a **separate parametric sigmoid only for selecting between MTJ integration and leakage**.

## Main Idea

Each MTJ neuron uses two different quantities derived from the synaptic input $z$:

1. A **normal sigmoid drive** that controls pulse strength.
2. A **parametric sigmoid gate** that determines how strongly the neuron follows integration versus leakage.

These two operations are intentionally separated.

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
MTJ Neuron
   |
   v
Linear(128, 64)
   |
   v
MTJ Neuron
   |
   v
Linear(64, 10)
   |
   v
MTJ Neuron
   |
   v
Spike Count over Time
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

## 1. Normal Sigmoid for Physical Drive

The original neural-input-to-drive mapping is kept unchanged:

$$d(z) = \sigma(\alpha z) = \frac{1}{1+\exp(-\alpha z)}$$

where:

- $z$ is the synaptic input,
- $\alpha$ is `input_scale`,
- $d(z)\in[0,1]$ is the normalized drive.

In the supplied configuration:

```python
input_scale = 1.0
```

so

$$d(z)=\sigma(z).$$

The drive is used to determine:

- current density,
- effective pulse width,
- MTJ integration dynamics.

It is **not** the integrate/leak selector in this version.

---

## 2. Parametric Sigmoid for Integrate/Leak Selection

A second sigmoid is introduced only for deciding whether the state update should favor integration or leakage:

$$g(z) = \sigma\left( k(z-z_0) \right) = \frac{1}{ 1+\exp[-k(z-z_0)] }.$$

The parameters are:

```python
GATE_SIGMOID_SLOPE = 1.0
GATE_SIGMOID_THRESHOLD = 0.0
```

where:

- $k$ is `GATE_SIGMOID_SLOPE`,
- $z_0$ is `GATE_SIGMOID_THRESHOLD`.

A larger $k$ produces a sharper transition:

$$k \uparrow \quad\Rightarrow\quad g(z) \text{ approaches a hard threshold.}$$

Approximately,

$$g(z)\approx \begin{cases} 0, & z<z_0,\\ 1, & z>z_0. \end{cases}$$

Useful values for a slope study include:

```text
1
5
10
25
50
```

---

## 3. Drive-to-Current Mapping

The normal sigmoid drive is mapped to the physical current-density range:

$$J = J_{\min} + d \left( J_{\max}-J_{\min} \right).$$

The implementation uses:

```python
J_MIN = 1e11
J_MAX = 1e12
```

therefore

$$10^{11} \le J \le 10^{12} \quad \mathrm{A/m^2}.$$

---

## 4. Current-Dependent Rise Time

The current-dependent rise-time constant is modeled as

$$\tau_r(J) = 386.98 \left( \frac{J}{10^{11}} \right)^{-1.223} + 8.88,$$

with time measured in picoseconds.

Higher current density produces a smaller rise-time constant and therefore faster integration.

---

## 5. MTJ Integration Model

The integration state is calculated using

$$m_{\mathrm{int}} = A(J) \tanh \left[ \frac{T_{\mathrm{eff}}}{\tau_r(J)} + \tanh^{-1} \left( \frac{m}{A(J)} \right) \right].$$

The effective pulse width is controlled by the **normal drive**:

$$T_{\mathrm{eff}} = d\,T_{\mathrm{pulse}}.$$

The default maximum pulse width is:

```python
PULSE_WIDTH_PS = 30.0
```

Therefore, the normal sigmoid affects both the current density and effective pulse width.

---

## 6. MTJ Leakage Model

The leakage factor is

$$B = \exp \left( -\frac{\Delta t}{\tau_l} \right).$$

The leakage state is

$$m_{\mathrm{leak}} = \frac{ mB }{ \sqrt{ 1-m^2(1-B^2) } }.$$

The default timing parameters are:

```python
DT_PS = 100.0
TAU_LEAK_PS = 503.8
```

---

## 7. Integrate/Leak State Update

The key modification in this version is that the **parametric sigmoid gate**, rather than the normal drive, controls the final mixing of integration and leakage:

$$\boxed{ m_{\mathrm{new}} = g(z)m_{\mathrm{int}} + \left[ 1-g(z) \right]m_{\mathrm{leak}} }$$

where

$$g(z) = \sigma \left( k(z-z_0) \right).$$

Therefore:

- $g\rightarrow1$: state follows mostly integration,
- $g\rightarrow0$: state follows mostly leakage.

The normal drive remains unchanged and is still used only in the physical pulse-strength calculations.

---

## 8. Special Case: Gate Slope = 1

If

```python
GATE_SIGMOID_SLOPE = 1.0
GATE_SIGMOID_THRESHOLD = 0.0
input_scale = 1.0
```

then

$$g(z)=\sigma(z)$$

and

$$d(z)=\sigma(z).$$

Thus,

$$g(z)=d(z).$$

In this case, the integrate/leak interpolation is mathematically the same as the previous implementation that directly used `drive`.

This is useful as a baseline sanity check before increasing the gate slope.

---

## 9. Spike Generation

The magnetization state acts as the membrane variable.

The firing threshold is:

```python
THRESHOLD = 0.8
```

Spike generation uses a snnTorch fast-sigmoid surrogate gradient:

```python
spike_grad = surrogate.fast_sigmoid(
    slope=25
)
```

The surrogate function enables backpropagation through the non-differentiable spike-generation operation.

---

## 10. No Hard Reset

The MTJ state is not reset to zero after a spike.

The updated state is retained and continues to evolve according to the nonlinear integration and leakage dynamics.

---

## 11. Classification

The output layer contains 10 spiking neurons.

Output spikes are summed across simulation time:

$$S_c = \sum_{t=1}^{T} s_c[t].$$

The predicted class is

$$\hat{y} = \arg\max_c S_c.$$

Cross-entropy loss is applied to the output spike counts.

---

## 12. Training Configuration

Default training settings are:

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

Gradient clipping is applied using:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    GRAD_CLIP
)
```

---

## 13. Dataset

The implementation uses MNIST from `torchvision.datasets`.

Input preprocessing is:

```python
transform = transforms.ToTensor()
```

so pixel values remain in the range

$$[0,1].$$

---

## 14. Firing-Rate Monitoring

The implementation reports the average firing rate of each spiking layer:

```text
Layer 1
Layer 2
Layer 3
```

during both training and testing.

The firing rate is calculated from the mean value of the binary spike tensor.

This enables direct comparison between:

- conventional LIF-based SNNs,
- the original smooth MTJ model,
- sharper parametric integrate/leak gating.

---

## Final Results

The final trained SMTJ model achieved:

```text
Best Test Accuracy: 96.99%

Final Test Accuracy: 96.99%

Final firing rates:
Layer 1 = 0.2233
Layer 2 = 0.4341
Layer 3 = 0.1614
```

### Performance Summary

| Metric | Result |
|---|---:|
| Best Test Accuracy | 96.99% |
| Final Test Accuracy | 96.99% |
| Layer 1 Firing Rate | 0.2233 |
| Layer 2 Firing Rate | 0.4341 |
| Layer 3 Firing Rate | 0.1614 |

The final classification accuracy is: $$96.99$$

## Main Hyperparameters

| Parameter | Default | Meaning |
|---|---:|---|
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `NUM_EPOCHS` | 10 | Training epochs |
| `NUM_STEPS` | 25 | SNN simulation steps |
| `THRESHOLD` | 0.8 | MTJ firing threshold |
| `PULSE_WIDTH_PS` | 30 ps | Maximum pulse width |
| `DT_PS` | 100 ps | Simulation/leak interval |
| `TAU_LEAK_PS` | 503.8 ps | Leakage time constant |
| `J_MIN` | $10^{11}$ A/m² | Minimum current density |
| `J_MAX` | $10^{12}$ A/m² | Maximum current density |
| `input_scale` | 1.0 | Normal drive sigmoid scale |
| `GATE_SIGMOID_SLOPE` | 1.0 | Integrate/leak gate steepness |
| `GATE_SIGMOID_THRESHOLD` | 0.0 | Integrate/leak transition point |
| `GRAD_CLIP` | 1.0 | Maximum gradient norm |

