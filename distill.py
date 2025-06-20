import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CIFAR10WithSoftLabels(Dataset):
    def __init__(self, train=True, transform=None, soft_labels=None):
        self.cifar = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, hard_label = self.cifar[idx]
        soft_label = self.soft_labels[idx]
        return image, hard_label, torch.tensor(soft_label, dtype=torch.float32)

def distillation_loss(student_logits, true_labels, teacher_soft, T, alpha, beta):
    ce_loss = F.cross_entropy(student_logits, true_labels)
    student_soft = F.log_softmax(student_logits/T, dim=1)
    teacher_soft = F.softmax(teacher_soft/T, dim=1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    return alpha * ce_loss + beta * (T **2) * kl_loss


def train_student(model, dataloader, optimizer, T=3.0, alpha=0.5, beta=0.5, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, hard_labels, soft_labels in dataloader:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = distillation_loss(outputs, hard_labels, soft_labels, T, alpha, beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {100*correct/total:.2f}%")

if __name__ == "__main__":
    # Load soft labels
    soft_labels = np.load('outputs/soft_labels.npy')

    # Transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset & Loader
    dataset = CIFAR10WithSoftLabels(train=False, transform=transform, soft_labels=soft_labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model
    student = StudentCNN().to(device)

    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    # Train
    train_student(student, dataloader, optimizer, T=3.0, alpha=0.5, beta=0.5, epochs=10)

    # Save
    os.makedirs("outputs", exist_ok=True)
    torch.save(student.state_dict(), "outputs/student_cnn.pth")
