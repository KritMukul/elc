import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json

def formatting_prompts_func(example):
    graph_str = json.dumps(example['graph_input'])
    spice_str = example['spice_output']

    text = (
        f"<|im_start|>system\nYou are an expert analog circuit designer. "
        f"Convert the provided netlist JSON graph representation into a valid, "
        f"simulatable SPICE netlist.<|im_end|>\n"
        f"<|im_start|>user\n{graph_str}<|im_end|>\n"
        f"<|im_start|>assistant\n{spice_str}<|im_end|>"
    )
    return text

def main():
    # 7B Coder model (You can comfortably fit the 14B or 32B on an H100 as well)
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct" 
    dataset_path = "master_parallel_dataset.json"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model: Native bfloat16 + Flash Attention 2 (No bitsandbytes!)
    print("Loading base model in native BF16 with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map="cuda", # Force it directly onto the H100
        trust_remote_code=True
    )

    # Setup LoRA Config (We can use a higher rank 'r' on an H100 for better learning)
    peft_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("Loading processed dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)

    # H100 Optimized Training Arguments
    training_args = SFTConfig(
        output_dir="./circuit_model_output",
        per_device_train_batch_size=8,      
        gradient_accumulation_steps=4,      
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,                          
        eval_strategy="steps",              
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        weight_decay=0.01,
        report_to="none", 
        dataloader_num_workers=4,
        max_length=2048                 # <--- MOVED HERE
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("Starting H100 training loop...")
    trainer.train()
    
    print("Saving fine-tuned adapter...")
    trainer.model.save_pretrained("./final_circuit_adapter")
    tokenizer.save_pretrained("./final_circuit_adapter")
    print("Complete!")

if __name__ == "__main__":
    main()