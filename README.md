# Fine-Tuning Falcon-7B with LoRA

## 📌 Overview
This repository demonstrates how to fine-tune the **Falcon-7B** model using **LoRA (Low-Rank Adaptation)** for efficient training. The training is performed on the **OpenAssistant-Guanaco dataset**.

## 🚀 Features
- Utilizes **LoRA** for memory-efficient fine-tuning
- Loads **Falcon-7B** with **4-bit quantization** using `BitsAndBytes`
- Uses **Hugging Face Transformers** for model and dataset handling
- Implements **SFTTrainer** for training

## 📦 Installation
Ensure you have Python 3.8+ and PyTorch installed. Then, install the required dependencies:

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes
```

## 📂 Dataset
This example fine-tunes the model on the **OpenAssistant-Guanaco** dataset:
```python
from datasets import load_dataset
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")
```

## 🔧 Model Configuration
The Falcon-7B model is loaded with 4-bit quantization:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "ybelkada/falcon-7b-sharded-bf16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

## 🔥 LoRA Configuration
We apply **LoRA** for efficient fine-tuning:
```python
from peft import LoraConfig

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)
```

## 🎯 Training Configuration
We define training parameters using `TrainingArguments`:
```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=200,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)
```

## 🚀 Training the Model
Fine-tune Falcon-7B using `SFTTrainer`:
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=200,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
```

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Feel free to contribute by submitting issues or pull requests.

## 📧 Contact
For any queries, reach out via [your email or GitHub profile].

