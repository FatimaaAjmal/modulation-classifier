"""
=============================================================================
 SNR SWEEP EXPERIMENT: Does AWGN noise degrade modulation classification?
=============================================================================
 PURPOSE:
   Test the modulation classifier at MANY different noise levels (SNRs)
   and measure how accuracy changes as the signal gets noisier.

 WHAT THIS ADDS to your original project:
   - Original: generated data at ONE fixed SNR (20 dB), trained, tested once.
   - This: trains on a MIX of SNRs (so the model is robust), then tests
     SEPARATELY at each SNR level and records the accuracy.
   - Output: an accuracy-vs-SNR curve showing noise degrading performance.

 This makes the claim true: "I tested it under AWGN across SNR levels
 and observed how accuracy falls off as noise increases."
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SIGNAL GENERATION (from your original generate_signals.py)
# ---------------------------------------------------------------------------
def generate_bpsk(n):
    bits = np.random.randint(0, 2, n)
    return (2 * bits - 1).astype(complex)

def generate_qpsk(n):
    bits = np.random.randint(0, 2, (n, 2))
    I = 2 * bits[:, 0] - 1
    Q = 2 * bits[:, 1] - 1
    return I + 1j * Q

def generate_16qam(n):
    bits = np.random.randint(0, 2, (n, 4))
    I = 2 * (2 * bits[:, 0] - 1) + (2 * bits[:, 1] - 1)
    Q = 2 * (2 * bits[:, 2] - 1) + (2 * bits[:, 3] - 1)
    return I + 1j * Q

def add_noise(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise

def make_dataset(n_per_class, snr_db):
    """Build a labelled dataset at a SPECIFIC snr_db."""
    data, labels = [], []
    gens = [generate_bpsk, generate_qpsk, generate_16qam]
    for label, gen in enumerate(gens):
        for _ in range(n_per_class):
            sig = gen(128)
            sig = add_noise(sig, snr_db)
            data.append([sig.real, sig.imag])
            labels.append(label)
    return (np.array(data, dtype=np.float32),
            np.array(labels, dtype=np.int64))

# ---------------------------------------------------------------------------
# THE MODEL (same architecture as your original train_model.py)
# ---------------------------------------------------------------------------
class ModulationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------------------
# STEP 1: TRAIN on a MIX of SNR levels (so the model isn't tuned to just one)
# ---------------------------------------------------------------------------
print("Building mixed-SNR training set...")
train_snrs = [-10, -5, 0, 5, 10, 15, 20]   # train across the whole range
X_list, y_list = [], []
for snr in train_snrs:
    Xs, ys = make_dataset(n_per_class=300, snr_db=snr)
    X_list.append(Xs); y_list.append(ys)
X_train = np.concatenate(X_list); y_train = np.concatenate(y_list)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=64, shuffle=True)

model = ModulationClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training...")
for epoch in range(25):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/25  loss {total_loss/len(train_loader):.3f}")

# ---------------------------------------------------------------------------
# STEP 2: TEST separately at each SNR level and record accuracy
# ---------------------------------------------------------------------------
print("\nTesting accuracy at each SNR level...")
test_snrs = [-10, -5, 0, 5, 10, 15, 20]
accuracies = []

model.eval()
for snr in test_snrs:
    Xt, yt = make_dataset(n_per_class=500, snr_db=snr)   # fresh test set at this SNR
    Xt_t = torch.tensor(Xt)
    yt_t = torch.tensor(yt)
    with torch.no_grad():
        preds = torch.max(model(Xt_t), 1)[1]
        acc = 100 * (preds == yt_t).sum().item() / len(yt_t)
    accuracies.append(acc)
    print(f"  SNR {snr:>4} dB  ->  accuracy {acc:5.1f}%")

# ---------------------------------------------------------------------------
# STEP 3: PLOT accuracy vs SNR (the key result)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(test_snrs, accuracies, 'o-', color='#1F4E79', linewidth=2, markersize=8)
plt.axhline(33.3, color='gray', linestyle='--', alpha=0.6,
            label='Random guess (33%)')
plt.xlabel('SNR (dB)  -  lower = noisier', fontsize=12)
plt.ylabel('Classification Accuracy (%)', fontsize=12)
plt.title('Modulation Classification Accuracy vs Noise Level (AWGN)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('accuracy_vs_snr.png', dpi=150)
print("\nSaved plot: accuracy_vs_snr.png")
print("\nSummary:")
for snr, acc in zip(test_snrs, accuracies):
    print(f"  {snr:>4} dB : {acc:.1f}%")
