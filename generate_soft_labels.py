import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

model = models.resnet50()
model.fc = torch.nn.Linear(2048, 10)
model.load_state_dict(torch.load("outputs/teacher_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=1)

def generate_soft_labels(model, dataloader, T=3.0, save_path="outputs/soft_labels.npy"):
    all_soft_labels = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            soft_targets = softmax_with_temperature(logits, T)
            all_soft_labels.append(soft_targets.cpu().numpy())

    soft_labels = np.concatenate(all_soft_labels, axis=0)
    np.save(save_path, soft_labels)
    print(f"Saved soft labels with T={T} to: {save_path}")

if __name__ == "__main__":
    generate_soft_labels(model, testloader, T=3.0)