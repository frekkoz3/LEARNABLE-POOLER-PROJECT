"""
    This material is develop for academic purpose. 
    It is develop by Francesco Bredariol as final project of the Introduction to ML course (year 2024-2025).
"""
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from poolers import *


class SimpleCNN(nn.Module):
    """
        This is a simple CNN with MaxPool2d.
        This architecture is implemented to work on mnist or cifar10 only.
    """
    def __init__(self, dataset = "MNIST"):
        """
            This architecture is developed only to work with the MNIST or the Cifar10 dataset. 
            If you are using the MNIST: dataset = "MNIST", if you are using cifar10: dataset = "cifar10".
            Other options will break the code.
        """
        super(SimpleCNN, self).__init__()

        if dataset == "cifar10":
            in_channels = 3
            last_level = 4
        if dataset == "MNIST":
            last_level = 3
            in_channels = 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 1, 28, 28) for MNIST,  (B, 3, 32, 32) for cifar10
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 16, 14, 14) for MNIST, (B, 16, 16, 16) for cifar10 

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 14, 14) for MNIST, (B, 32, 16, 16) for cifar10
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 32, 7, 7) for MNIST, (B, 32, 8, 8) for cifar10

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * last_level * last_level, 128) # 64x3x3 for MNIST, 64x4x4 for cifar10
        self.fc2 = nn.Linear(128, 10)

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
    
    def get_core(self):
        return "No core."
    
    def name(self):
        return "Simple CNN"

class TDSGSimpleCNN(nn.Module):
    """
            This is a simple CNN with TDGSPooling2d.
            This architecture works exactly only on cifar10 or mnist
        """
    def __init__(self, dataset = "cifar10", initial_value = 1):
        """
            This architecture is developed only to work with the MNIST or the Cifar10 dataset. 
            If you are using the MNIST: dataset = "MNIST", if you are using cifar10: dataset = "cifar10".
            Other options will break the code.
        """
        super(TDSGSimpleCNN, self).__init__()

        if dataset == "MNIST":

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 1, 28, 28)
            
            self.pool1 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 16, device = "cuda", H_out = 14, W_out = 14, initial_value=initial_value)  # Output: (B, 16, 14, 14) for MNIST, (B, 16, 16, 16) for cifar10 

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 14, 14) for MNIST, (B, 32, 16, 16) for cifar10
            
            self.pool2 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 32, device = "cuda", H_out = 7, W_out = 7, initial_value=initial_value)  # Output: (B, 32, 7, 7) for MNIST, (B, 32, 8, 8) for cifar10

            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (B, 64, 7, 7) for MNIST, (B, 64, 8, 8) for cifar10
            
            self.pool3 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 64, device = "cuda", H_out = 3, W_out = 3, initial_value=initial_value)  # Output: (B, 64, 3, 3) for MNIST, (B, 64, 4, 4) for cifar10
            
            self.fc1 = nn.Linear(64 *3 * 3 , 128)

        if dataset == "cifar10":
        
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # Input: (B, 3, 32, 32)
            
            self.pool1 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 16, device = "cuda", H_out = 16, W_out = 16, initial_value=initial_value)  # Output: (B, 16, 14, 14) for MNIST, (B, 16, 16, 16) for cifar10 

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (B, 32, 14, 14) for MNIST, (B, 32, 16, 16) for cifar10
            
            self.pool2 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 32, device = "cuda", H_out = 8, W_out = 8, initial_value=initial_value)  # Output: (B, 32, 7, 7) for MNIST, (B, 32, 8, 8) for cifar10

            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: (B, 64, 7, 7) for MNIST, (B, 64, 8, 8) for cifar10
            
            self.pool3 = TDGSPooling2d(kernel_size=2, stride=2, in_channels = 64, device = "cuda", H_out = 4, W_out = 4, initial_value=initial_value)  # Output: (B, 64, 3, 3) for MNIST, (B, 64, 4, 4) for cifar10
            
            self.fc1 = nn.Linear(64 * 4 * 4 , 128)

        self.fc2 = nn.Linear(128, 10)

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
    
    def get_core(self):
        s = f"FIRST POOLING {self.pool1.get_core()}\nSECOND POOLING {self.pool2.get_core()}\nTHIRD POOLING {self.pool3.get_core()}"
        return s
    
    def name(self):
        return f"TDGS Simple CNN - Initialization at t = {self.pool1.initial_value}"
    
def training(train_loader, model, device, criterion, optimizer, epochs = 10, verbose = True):
    train_losses = []
    for epoch in range(epochs):  # number of epochs
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
        train_losses.append(total_loss)
        if verbose :
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    return train_losses

def testing(test_loader, model, device):
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
    return (100*correct)/total # Return the accuracy

def history_plot(history, model_name):
    
    epochs = [i for i in range (len(history))]

    plt.plot(epochs, history, label=model_name, color = "blue")
    plt.scatter(epochs, history, color = "blue", marker = "x")

    plt.title(model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.tight_layout()
    plt.show()

def comparison_plot(container):

    for single_model_data in container:
        h, n, c = single_model_data
        epochs = [i for i in range (len(h))]

        plt.plot(epochs, h, label=n, color = c)
        plt.scatter(epochs, h, color = c, marker = "x")

    plt.title("Loss histories comparisation")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.tight_layout()
    plt.show()

def performances_plot(initial_values, performances, dataset_name):
    plt.plot(initial_values, performances, color = 'r')
    plt.scatter(initial_values, performances, marker='o', color = 'k', s = 40)
    plt.title(f"Accuracy wrt initial value of T - {dataset_name}")
    plt.xlabel("Initial value of T")
    plt.ylabel("Test Accuracy %")
    plt.tight_layout()
    plt.show()

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
    model = TDSGSimpleCNN(dataset="cifar10").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = training(train_loader, model, device, criterion, optimizer, epochs=10)
    accuracy = testing(test_loader, model, device)

    history_plot(history, model.name())

    print(model.get_core())


