from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import json

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def load_data(filepath):
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            input_text = data["input"]
            target_text = json.dumps(data["expected_output"])
            examples.append({"input_text": input_text, "target_text": target_text})
    return examples

train_dataset = load_data("function_call_training_data.jsonl")

def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], truncation=True)
    targets = tokenizer(examples["target_text"], truncation=True)
    return {"input_ids": inputs["input_ids"], "labels": targets["input_ids"]}

from datasets import Dataset
dataset = Dataset.from_list(train_dataset)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_function_call_model")
tokenizer.save_pretrained("./fine_tuned_function_call_model")
