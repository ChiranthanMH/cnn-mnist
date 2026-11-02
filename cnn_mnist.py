import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST
print("Loading MNIST...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers (learn patterns)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 32 filters, 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filters, 3x3
        
        # Pooling (reduce image size)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dense layers (classification)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2 pooling: 28->14->7
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("Training CNN...\n")
epochs = 3
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Test Accuracy: {accuracy:.4f}")

print(f"\nFinal CNN Accuracy: {test_accuracies[-1]:.4f}")

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Test Accuracy')
plt.grid()

plt.tight_layout()
plt.savefig('cnn_training.png')
plt.show()