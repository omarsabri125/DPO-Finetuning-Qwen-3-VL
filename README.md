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
- [Usage](#-usage)
- [Configuration](#-configuration)
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
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install torch torchvision transformers datasets
pip install llamafactory

# Or install from requirements file
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

This step downloads the HuggingFace dataset, processes images as JPEGs, and converts everything to LlamaFactory's ShareGPT DPO format.

### 2. Register the Dataset

Update `data/dataset_info.json` to include your custom dataset entry, pointing to the formatted data file.

### 3. Configure Training

Edit `configs/dpo_config.yaml` to set your desired hyperparameters (see [Configuration](#-configuration) below).

### 4. Run DPO Training

```bash
llamafactory-cli train configs/dpo_config.yaml
```

---

## 🔧 Configuration

Training is controlled via a YAML configuration file. Key parameters include:

```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
stage: dpo
do_train: true
finetuning_type: lora

dataset: topic_overwrite
template: qwen3_vl

lora_rank: 8
lora_target: all

per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0

bf16: true
output_dir: outputs/qwen3-vl-dpo
```

---

## 📚 References

- [Direct Preference Optimization (DPO) Paper](https://arxiv.org/abs/2305.18290)
- [LlamaFactory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [helehan/topic-overwrite Dataset](https://huggingface.co/datasets/helehan/topic-overwrite)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
