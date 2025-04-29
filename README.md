# Efficient Fine-Tuning with NdLinear Layers

This project explores efficient model compression by replacing standard Linear layers with **NdLinear** layers in a lightweight Vision Transformer (SmallViT) architecture.

We benchmarked and compared performance between the baseline model (standard Linear) and the modified NdLinear model across 5 epochs, focusing on maintaining accuracy while reducing parameter overhead.

---

## 📚 Project Summary

| Model Variant       | # Parameters | Accuracy (5 Epochs) |
|---------------------|--------------|---------------------|
| Baseline (Linear)    | 5.52M        | 95.46%              |
| NdLinear             | 5.52M (slightly fewer) | 95.58%         |

- **NdLinear layers** achieve a **minor compression benefit** without sacrificing accuracy.
- Demonstrates the **potential of efficient fine-tuning strategies** for transformer-based models.
- Lays the groundwork for **further compression** techniques such as combining **NdLinear with LoRA adapters**.

---

## 🛠️ Project Structure

- `Aryan_EnesembleAI_Submission.ipynb` — Main notebook containing:
  - Model definitions for Baseline (Linear) and NdLinear variants.
  - Training and evaluation scripts.
  - Benchmark results with visualizations.
  - Future work and conclusions.

---

## 🚀 Key Features

- Replaced standard `nn.Linear` layers with custom `NdLinear` layers.
- Maintained model expressiveness while achieving slight parameter savings.
- Benchmarked model performance over 5 epochs on the **CIFAR-10** dataset.
- Compared training loss and accuracy curves between the two models.

---

## 📈 Results

- **Accuracy**: NdLinear model slightly outperforms the baseline after 5 epochs.
- **Parameters**: NdLinear achieves a **parameter-efficient representation** while preserving learning capabilities.

---

## 🧠 Future Work

- Integrate **NdLinear** layers with **LoRA adapters** to achieve even greater fine-tuning efficiency.
- Extend experiments to larger datasets such as **CIFAR-100** or **ImageNet**.
- Analyze compression effects on model generalization and downstream tasks.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 📝 Citation

If you find this work helpful, feel free to cite or mention this repository!

---

## ✨ Acknowledgements

This project was prepared as part of the EnsembleAI Research Internship Application process, focusing on innovations in **efficient fine-tuning and model compression**.

---
```

---

Would you also like me to prepare:
- A `LICENSE` file (MIT License standard for open projects)?
- A small `requirements.txt` file for dependencies?
- A cleaner GitHub folder structure suggestion (e.g., move code to `src/`, results to `results/`)?

Would you like those too? 🚀
