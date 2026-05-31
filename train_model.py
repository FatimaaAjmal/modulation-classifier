import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load dataset
dataset = np.load('dataset.npy')
labels = np.load('labels.npy')

# Convert to PyTorch tensors
X = torch.tensor(dataset, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# Split into training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Build the neural network
class ModulationClassifier(nn.Module):
    def __init__(self):
        super(ModulationClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model, loss function and optimizer
model = ModulationClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = []
test_accuracies = []
epochs = 20

print("Training started...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, color='blue')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(test_accuracies, color='green')
ax2.set_title('Classification Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.suptitle('Automatic Modulation Classification - BPSK vs QPSK vs 16QAM')
plt.tight_layout()
plt.savefig('results.png')
plt.show()
print("Results saved as results.png")