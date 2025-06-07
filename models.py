"""
    This material is develop for academic purpose. 
    It is develop by Francesco Bredariol as final project of the Introduction to ML course (year 2024-2025).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from poolers import *

class SimpleCNN(nn.Module):
    """
        This is a simple CNN with MaxPool2d.
        This architecture should work for the MNIST dataset.
    """
    def __init__(self, num_classes=10, in_channels = 3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 1, 28, 28) for MNIST,  (B, 3, 32, 32) for cifar10
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 16, 14, 14) for MNIST, (B, 16, 16, 16) for cifar10 

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 14, 14) for MNIST, (B, 32, 16, 16) for cifar10
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 32, 7, 7) for MNIST, (B, 32, 8, 8) for cifar10

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 4 * 4 , 128) # 64x3x3 for MNIST, 64x4x4 for cifar10
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TDSGSimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels = 3):
        """
            This is a simple CNN with TDGSPooling2d.
            This architecture should work for the MNIST dataset.
        """
        super(TDSGSimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 3, 28, 28)
        
        self.pool1 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 16, device = "cuda", H_out = 16, W_out = 16)  # Output: (B, 16, 14, 14) for MNIST, (B, 16, 16, 16) for cifar10 

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 14, 14) for MNIST, (B, 32, 16, 16) for cifar10
        
        self.pool2 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 32, device = "cuda", H_out = 8, W_out = 8)  # Output: (B, 32, 7, 7) for MNIST, (B, 32, 8, 8) for cifar10

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (B, 64, 7, 7) for MNIST, (B, 64, 8, 8) for cifar10
        
        self.pool3 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 64, device = "cuda", H_out = 4, W_out = 4)  # Output: (B, 64, 3, 3) for MNIST, (B, 64, 4, 4) for cifar10
        
        self.fc1 = nn.Linear(64 * 4 * 4 , 128) # It is 8x8 for SP3, 7x7 for the others
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64)

    # Initialize model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = TDSGSimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model.pool1.get_core())
    print(model.pool2.get_core())
    print(model.pool3.get_core())

    for epoch in range(20):  # number of epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    print(model.pool1.get_core())
    print(model.pool2.get_core())
    print(model.pool3.get_core())


