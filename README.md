# 🦙 DPO Fine-Tuning with LlamaFactory — Qwen3-VL-4B Vision-Language Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LlamaFactory](https://img.shields.io/badge/LlamaFactory-latest-orange)](https://github.com/hiyouga/LLaMA-Factory)
[![Model](https://img.shields.io/badge/Model-Qwen3--VL--4B-purple)](https://huggingface.co/Qwen)
[![Training](https://img.shields.io/badge/Method-DPO-green)](https://arxiv.org/abs/2305.18290)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

A complete **Direct Preference Optimization (DPO)** fine-tuning pipeline for a **Vision-Language Model (VLM)** using the **LlamaFactory** framework, targeting **Qwen3-VL-4B-Instruct** on image-question preference data.

---

## 📖 Table of Contents

- [What is DPO?](#-what-is-dpo)
- [What is LlamaFactory?](#-what-is-llamafactory)
- [Model](#-model)
- [Dataset](#-dataset)
- [Pipeline Overview](#️-pipeline-overview)
- [Installation](#-installation)
- [References](#-references)

---

## 🎯 What is DPO?

**Direct Preference Optimization (DPO)** is a fine-tuning technique that teaches a language model to prefer certain responses over others — without needing a separate reward model.

Instead of reinforcement learning from human feedback (RLHF), DPO directly optimizes the model using **preference pairs**:

| Label | Description |
|---|---|
| ✅ **Chosen** | The preferred, higher-quality response |
| ❌ **Rejected** | The non-preferred, lower-quality response |

DPO is simpler, more stable, and computationally cheaper than PPO-based RLHF while achieving comparable alignment quality.

> 📄 Original paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

---

## 🏭 What is LlamaFactory?

**LlamaFactory** is an open-source, unified fine-tuning framework that supports a wide range of training methods and model families.

**Supported training methods:** SFT, DPO, ORPO, KTO, PPO, and more  
**Supported adapters:** LoRA, QLoRA, full fine-tuning  
**Supported modalities:** Text, images, audio, video  
**Supported model families:** LLaMA, Qwen, Mistral, Gemma, and many more

> 🔗 [LlamaFactory GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)

---

## 🤖 Model

We fine-tune **[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)** — a 4-billion parameter vision-language model from Alibaba's Qwen3 family, capable of understanding both text and images.

| Property | Value |
|---|---|
| Parameters | 4B |
| Modalities | Text + Images |
| Family | Qwen3-VL |
| Developer | Alibaba |

---

## 📦 Dataset

We use the **[`helehan/topic-overwrite`](https://huggingface.co/datasets/helehan/topic-overwrite)** dataset from HuggingFace.

This dataset contains image-question pairs with **chosen** and **rejected** answers, making it ideal for DPO preference training on a vision-language task.

---

## 🗺️ Pipeline Overview

```
1. Load Dataset          →  HuggingFace dataset with images + chosen/rejected answers
2. Process Images        →  Save as JPEG, map paths back to DataFrame
3. Format for DPO        →  Convert to LlamaFactory's ShareGPT DPO format
4. Download Extra Data   →  Pull pre-prepared files from Google Drive
5. Fix Image Paths       →  Remap paths from Colab → Kaggle
6. Install Dependencies  →  torch, transformers, LlamaFactory
7. Register Datasets     →  Add custom datasets to LlamaFactory registry
8. Write YAML Config     →  Define all training hyperparameters
9. Run Training          →  Launch DPO training via CLI
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/omarsabri125/DPO-Finetuning-Qwen-3-VL.git
cd DPO-Finetuning-Qwen-3-VL

```

## 📚 References

- [Direct Preference Optimization (DPO) Paper](https://arxiv.org/abs/2305.18290)
- [LlamaFactory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [helehan/topic-overwrite Dataset](https://huggingface.co/datasets/helehan/topic-overwrite)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
