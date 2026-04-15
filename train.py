import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_name = "Qwen/Qwen2.5-0.5B"
dataset_path = "dataset.json"
max_length = 64


def load_json_dataset(path: str) -> Dataset:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    with path_obj.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Format dataset harus berupa list of objects (JSON array).")

    formatted_data = []
    for i, item in enumerate(raw_data):
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()

        if not instruction or not output:
            continue

        text = f"Instruction: {instruction}\nResponse: {output}"
        formatted_data.append({"text": text})

    if not formatted_data:
        raise ValueError("Dataset kosong atau tidak memiliki field 'instruction' dan 'output' yang valid.")

    print(f"Total data valid: {len(formatted_data)}")
    print("Contoh data:")
    print(formatted_data[0]["text"])

    return Dataset.from_list(formatted_data)


def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Load tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Load dataset dari file JSON
dataset = load_json_dataset(dataset_path)
dataset = dataset.select(range(3))
dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Konfigurasi LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./model-lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./model-lora")
tokenizer.save_pretrained("./model-lora")

print("done")
