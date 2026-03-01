# Gradient Modulator

The gradient modulator $\Psi_\phi$ is a **ConvLSTM** network that processes
the gradient signal and current state to produce a refined gradient update.

## Architecture

Input: $[\mathbf{g}^{(k)}, \mathbf{x}^{(k-1)}]$ concatenated along the
channel axis, plus the previous LSTM hidden state $\mathbf{s}^{(k-1)}$.

ConvLSTM gates (for 1-D, convolutions over the spatial axis $N$):

$$\mathbf{i} = \sigma(\mathbf{W}_{xi} * \mathbf{x}_{in} + \mathbf{W}_{hi} * \mathbf{h})$$
$$\mathbf{f} = \sigma(\mathbf{W}_{xf} * \mathbf{x}_{in} + \mathbf{W}_{hf} * \mathbf{h})$$
$$\mathbf{g} = \tanh(\mathbf{W}_{xg} * \mathbf{x}_{in} + \mathbf{W}_{hg} * \mathbf{h})$$
$$\mathbf{o} = \sigma(\mathbf{W}_{xo} * \mathbf{x}_{in} + \mathbf{W}_{ho} * \mathbf{h})$$
$$\mathbf{c}' = \mathbf{f} \odot \mathbf{c} + \mathbf{i} \odot \mathbf{g}$$
$$\mathbf{h}' = \mathbf{o} \odot \tanh(\mathbf{c}')$$

Output: Projected back to the state space via a final convolution.

## fourdvarjax Implementation

- `ConvLSTMGradMod1D` — 1-D spatial convolutions
- `ConvLSTMGradMod2D` — 2-D spatial convolutions
