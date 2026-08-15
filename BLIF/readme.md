# MNIST Spiking MLP with Standard LIF Neurons

This project implements a spiking neural network (SNN) version of the MNIST multilayer perceptron architecture:

```text
784 -> 128 -> 64 -> 10
```

The network uses standard **Leaky Integrate-and-Fire (LIF)** neurons from `snnTorch` and keeps the same fully connected structure as the original MLP.

The purpose of this implementation is to provide a conventional LIF baseline that can be compared directly with the MTJ-inspired neuron models using the same architecture and reporting format.

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
LIF
   |
   v
Linear(128, 64)
   |
   v
LIF
   |
   v
Linear(64, 10)
   |
   v
LIF
   |
   v
Temporal output
   |
   v
Predicted digit
```

The network is simulated for:

```python
num_steps = 25
```

time steps.

---

## LIF Neuron Model

Each spiking layer uses:

```python
snn.Leaky(
    beta=beta,
    spike_grad=spike_grad,
    init_hidden=False
)
```

with:

```python
beta = 0.9
```

The parameter $\beta$ controls membrane-potential decay between time steps.

Conceptually, a discrete-time LIF neuron follows a recurrence of the form

$$U[t+1] = \beta U[t] + I[t+1] - R[t]$$

where:

- $U[t]$ is the membrane potential,
- $I[t]$ is the synaptic input current,
- $\beta$ is the leak/decay coefficient,
- $R[t]$ represents the reset contribution after spiking.

The exact reset behavior is handled internally by `snn.Leaky`.

---

## Spike Generation

A neuron emits a spike when its membrane potential crosses the firing threshold.

Because the hard spike function is non-differentiable, the implementation uses a fast-sigmoid surrogate gradient:

```python
spike_grad = surrogate.fast_sigmoid(
    slope=25
)
```

The surrogate is used during backpropagation while the forward spike remains binary.

---

## Dataset

The code uses the MNIST dataset from `torchvision.datasets`.

Input preprocessing is:

```python
transform = transforms.ToTensor()
```

Therefore, image pixels remain in the range

$$[0,1].$$

No additional normalization is applied.

---

## Network Layers

### Layer 1

```python
self.fc1 = nn.Linear(
    28 * 28,
    128
)
```

followed by:

```python
self.lif1 = snn.Leaky(...)
```

### Layer 2

```python
self.fc2 = nn.Linear(
    128,
    64
)
```

followed by:

```python
self.lif2 = snn.Leaky(...)
```

### Output Layer

```python
self.fc3 = nn.Linear(
    64,
    10
)
```

followed by:

```python
self.lif3 = snn.Leaky(...)
```

---

## Temporal Simulation

For every MNIST sample, the same input is processed for multiple simulation steps:

```python
for step in range(num_steps):
```

At each time step:

1. The image is passed through the first fully connected layer.
2. The first LIF layer updates its membrane and emits spikes.
3. Those spikes are passed to the second fully connected layer.
4. The second LIF layer updates its membrane and emits spikes.
5. Those spikes are passed to the output layer.
6. Output spikes and membrane potentials are recorded.

The recorded spike tensors have the form:

```text
[time, batch, neurons]
```

---

## Training Output

For training, the implementation sums the output-layer membrane potentials over time:

$$O = \sum_{t=1}^{T} U_3[t]$$

In code:

```python
output = mem3.sum(
    dim=0
)
```

This produces a tensor of shape:

```text
[batch, 10]
```

which is used with cross-entropy loss:

```python
criterion = nn.CrossEntropyLoss()
```

The predicted class is:

$$\hat{y} = \arg\max_c O_c$$

---

## Optimizer

The implementation uses SGD:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate
)
```

with:

```python
learning_rate = 0.01
```

This preserves the optimizer choice of the original LIF baseline.

---

## Training Configuration

Default hyperparameters are:

```python
batch_size = 64
learning_rate = 0.01
num_epochs = 10
num_steps = 25
beta = 0.9
```

---

## Firing-Rate Monitoring

The implementation records spikes from all three LIF layers:

```python
spk1_rec
spk2_rec
spk3_rec
```

The firing rate is computed as the mean of each binary spike tensor:

$$FR = \frac{\text{number of spikes}}{\text{number of neuron-time events}}$$

The script reports firing rates for:

```text
Layer 1
Layer 2
Layer 3
```

during both training and testing.

This makes the baseline directly comparable with the MTJ-inspired SNN implementations.

---

## Training Output Format

During training, intermediate output is printed in the form:

```text
Epoch 1 | Batch    0/938 | Loss 2.3026 | Spike FR: 0.100, 0.050, 0.020
```

At the end of each epoch:

```text
---------------- TRAIN ----------------

Loss:
Accuracy:

Output spike firing rates:
Layer 1:
Layer 2:
Layer 3:

---------------- TEST ----------------

Accuracy:

Output spike firing rates:
Layer 1:
Layer 2:
Layer 3:
```

---

## Best-Model Saving

Whenever test accuracy improves, the network is saved as:

```text
best_mnist_lif_mlp.pth
```

After training, the best checkpoint is reloaded and evaluated again.

---

## Final Membrane-Based Accuracy

The main test accuracy uses summed output membrane potentials:

$$O = \sum_t U_3[t]$$

The predicted class is obtained from:

$$\arg\max_c O_c$$

This is the same criterion used during training.

---

## Final Spike-Count Accuracy

The implementation additionally evaluates classification using only the number of output spikes.

For each class:

$$S_c = \sum_{t=1}^{T} s_c[t]$$

The predicted class becomes:

$$\hat{y} = \arg\max_c S_c$$

In code:

```python
spike_count = spk_rec.sum(
    dim=0
)

predicted = spike_count.argmax(
    dim=1
)
```

This provides a second evaluation metric that uses purely spiking output activity rather than membrane potential.

---

## Final Results

The final trained BLIF model achieved:

```text
====================================================
Training complete
====================================================
Best Test Accuracy: 96.93%

Final Test Accuracy: 96.93%

Final output firing rates:
Layer 1 = 0.0874
Layer 2 = 0.0492
Layer 3 = 0.0872

Spike-count Test Accuracy: 97.12%

Model saved as mnist_snn_mlp.pth
```

### Performance Summary

| Metric | Result |
|---|---:|
| Best Test Accuracy | 96.93% |
| Final Test Accuracy | 96.93% |
| Spike-count Test Accuracy | 97.12% |
| Layer 1 Firing Rate | 0.0874 |
| Layer 2 Firing Rate | 0.0492 |
| Layer 3 Firing Rate | 0.0872 |

The spike-count evaluation gives the highest reported accuracy:

$$\boxed{97.12\%}$$

## Main Hyperparameters

| Parameter | Default | Meaning |
|---|---:|---|
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 0.01 | SGD learning rate |
| `num_epochs` | 10 | Number of training epochs |
| `num_steps` | 25 | Number of SNN simulation steps |
| `beta` | 0.9 | LIF membrane decay |
| surrogate slope | 25 | Fast-sigmoid surrogate steepness |

---

## Comparison Baseline

This implementation is intended to serve as the conventional LIF baseline for comparison against alternative neuron models.

Because the architecture is kept fixed,

```text
784 -> 128 -> 64 -> 10
```

differences in:

- classification accuracy,
- spike-count accuracy,
- firing rate,
- temporal activity,

can be compared more directly with MTJ-inspired neuron implementations.

The important distinction is that this model uses the standard `snnTorch` LIF dynamics rather than the nonlinear MTJ integration and leakage equations.

---

## Summary

The overall model can be summarized as:

$$x \rightarrow \mathrm{FC}_1 \rightarrow \mathrm{LIF}_1 \rightarrow \mathrm{FC}_2 \rightarrow \mathrm{LIF}_2 \rightarrow \mathrm{FC}_3 \rightarrow \mathrm{LIF}_3$$

The network is trained using temporally summed output membrane potentials:

$$\boxed{O_c = \sum_{t=1}^{T} U_c[t]}$$

and is additionally evaluated using output spike counts:

$$\boxed{ S_c = \sum_{t=1}^{T} s_c[t] }$$

This provides a straightforward conventional SNN baseline for comparison with the MTJ-based neuron models.
