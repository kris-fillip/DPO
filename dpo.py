import os
import torch
import json

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
import bitsandbytes as bnb
from huggingface_hub import login


def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    print(list(lora_module_names))
    return list(lora_module_names)
    
login()

model_name = "Kris-Fillip/llama_base_sft"
output_dir="./results_dpo"
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

train_dataset = load_dataset("Kris-Fillip/reddit_train", split="train")

validation_dataset = load_dataset("Kris-Fillip/reddit_validation",split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir="llama_model_7b"
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def return_prompt_and_responses(samples):
    return {
        "prompt": samples["prompt"],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }
print("Formatting data..")
original_train_columns = train_dataset.column_names

train_dataset = train_dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_train_columns
)

original_validation_columns = validation_dataset.column_names

validation_dataset = validation_dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_validation_columns
)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    max_grad_norm= 0.3,
    num_train_epochs=1,
    max_steps=-1, 
    save_steps=100,
    eval_steps=100,
    learning_rate=5e-4,
    bf16=True,
    save_total_limit=5,
    logging_steps=50,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    dataloader_drop_last=True,
    auto_find_batch_size=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    }
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=1024,
    
)
print_trainable_parameters(model)

print("Starting training..")
dpo_trainer.train()

dpo_trainer.save_model(final_checkpoint_dir)
print("Saved model!")

# Load the entire model on the GPU 0
device_map = {"": 0}
reloaded_model = AutoPeftModelForCausalLM.from_pretrained(
    final_checkpoint_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    cache_dir="llama_model_7b"
)
reloaded_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir, add_eos_token=True, use_fast=True)
print("Reloaded Model!")
# Merge the LoRA and the base model
merged_model = reloaded_model.merge_and_unload()
# Save the merged model
merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
merged_model.save_pretrained(merged_dir)
reloaded_tokenizer.save_pretrained(merged_dir)
print("Saved Merged checkpoint!")