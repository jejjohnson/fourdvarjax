# Algorithm Pseudocode

```
Algorithm: 4DVarNet Training

Input:
  - Dataset D = {(y_i, x*_i)} of (observation, ground-truth) pairs
  - Prior model φ_θ (bilinear autoencoder)
  - Gradient modulator Ψ_φ (ConvLSTM)
  - Number of solver steps K
  - Learning rate η

Initialise θ, φ randomly

For each epoch:
  For each batch (y, x*) in D:
    # === Forward pass (unrolled solver) ===
    x ← m ⊙ y                           # Initialise with masked obs
    s ← 0                                 # Zero LSTM state
    For k = 1, ..., K:
      g ← ∇_x [||m⊙(x−y)||² + λ||x−φ_θ(x)||²]
      d, s ← Ψ_φ(g, x, s)
      x ← x − α·d

    # === Loss and backward ===
    L ← ||x − x*||²
    (θ, φ) ← (θ, φ) − η · ∇_(θ,φ) L

Return θ, φ
```
