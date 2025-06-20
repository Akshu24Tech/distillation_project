import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from models.student import StudentCNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.student import StudentCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

def evaluate_model(model, name="Model"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{name} Accuracy: {accuracy:.2f}%")
    return accuracy

def load_models():
    # Load Teacher
    teacher = models.resnet50()
    teacher.fc = torch.nn.Linear(2048, 10)
    teacher.load_state_dict(torch.load("outputs/teacher_resnet50.pth", map_location=device))
    teacher = teacher.to(device)

    # Load Student
    student = StudentCNN().to(device)
    student.load_state_dict(torch.load("outputs/student_cnn.pth", map_location=device))

    return teacher, student

if __name__ == "__main__":
    teacher, student = load_models()
    acc_teacher = evaluate_model(teacher, name="Teacher (ResNet-50)")
    acc_student = evaluate_model(student, name="Student (Custom CNN)")

    print("\n📊 Final Comparison:")
    print(f"Teacher Accuracy : {acc_teacher:.2f}%")
    print(f"Student Accuracy : {acc_student:.2f}%")
