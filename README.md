# 🧠 Knowledge Distillation from Neural Networks

A PyTorch implementation of the seminal research paper:

> **Distilling the Knowledge in a Neural Network**  
> Geoffrey Hinton, Oriol Vinyals, Jeff Dean — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

This project demonstrates how to transfer knowledge from a large, high-performing **Teacher model** to a smaller, efficient **Student model** using **soft targets** and **temperature-scaled outputs**.

---

## 📘 What is Knowledge Distillation?

Knowledge distillation is a compression technique where a small model (student) is trained to replicate the behavior of a large model (teacher). Rather than training the student on hard labels (like `[0, 0, 1]`), it learns from the **soft probability distributions** output by the teacher. This allows the student to mimic the generalization capability of the teacher while being faster and lighter.

---

## 🔬 Research Paper Summary

- Teacher: Large network trained with ground truth
- Student: Smaller model trained using:
  - Hard labels (`CrossEntropy`)
  - Soft targets from the teacher (`KL Divergence`)
- Temperature `T` is used to soften the teacher outputs:
  \[
  \text{Loss} = \alpha \cdot \text{CE} + \beta \cdot T^2 \cdot \text{KL}(\text{soft}_T^{\text{teacher}} || \text{soft}_T^{\text{student}})
  \]

---

## 📂 Project Structure

```

distillation\_project/
│
├── models/                   # Teacher and Student architectures
│   └── student.py
│
├── utils/                    # Training and evaluation scripts
│   ├── train\_teacher.py
│   ├── generate\_soft\_labels.py
│   └── evaluate.py
│
├── outputs/                  # Model weights and soft label dumps
│   ├── teacher\_resnet50.pth
│   ├── student\_cnn.pth
│   └── soft\_labels.npy
│
├── distill.py                # Main distillation training script
├── requirements.txt
└── README.md

````

---

## 🚀 How to Run

### 1. 🔧 Install Requirements

```bash
pip install -r requirements.txt
````

### 2. 🏋️‍♂️ Train the Teacher Model (ResNet-50 on CIFAR-10)

```bash
python utils/train_teacher.py
```

### 3. 🔥 Generate Soft Labels using the Teacher

```bash
python utils/generate_soft_labels.py
```

### 4. 👨‍🏫 Train the Student Model with Distillation

```bash
python distill.py
```

### 5. 📊 Evaluate Both Models

```bash
python utils/evaluate.py
```

---

## 📈 Results

| Model                | Accuracy (%) | Parameters | Size (MB) |
| -------------------- | ------------ | ---------- | --------- |
| Teacher (ResNet-50)  | \~93%        | 23M        | \~98 MB   |
| Student (Custom CNN) | \~88%        | <1M        | \~3 MB    |

> ✅ Significant reduction in size with minimal accuracy drop.

---

## 🧪 Datasets

* CIFAR-10 — 60,000 32x32 color images in 10 classes
* Automatically downloaded via `torchvision.datasets.CIFAR10`

---

## 📌 Techniques Used

* Knowledge Distillation
* Soft Targets with Temperature Scaling
* KL Divergence
* PyTorch + torchvision
* Model compression

---

## 📚 References

* [Distilling the Knowledge in a Neural Network (arXiv:1503.02531)](https://arxiv.org/abs/1503.02531)
* Hinton et al., 2015
* PyTorch Tutorials

---

## 🛠️ Future Work

* Replace Student CNN with ResNet-18 or MobileNet
* Visualize soft targets and class probabilities
* Apply to other datasets (e.g., ImageNet, TinyImageNet)
* Deploy with ONNX or TensorRT for inference

---

## 🙌 Acknowledgements

Built as part of an educational implementation of classic AI research.
Thanks to [Geoffrey Hinton](https://scholar.google.com/citations?user=JicYPdAAAAAJ&hl=en) and the original authors for their foundational work.

---

## 📌 License

This project is open-source for educational and research purposes.
