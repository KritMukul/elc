import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def main():
    print("Loading SchGen dataset...")
    # Load the JSONL file you just downloaded
    dataset = load_dataset("microsoft/SchGen_dataset", split="train")    
    # We will use Qwen2.5-Coder-1.5B-Instruct for prototyping (it fits on most GPUs).
    # You can change this to "Qwen/Qwen2.5-Coder-7B-Instruct" for production.
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Qwen models require pad_token to be set for fine-tuning
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 # Use bfloat16 for Ampere/Hopper GPUs (like DGX)
    )
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Format the messages into a single prompt string using the model's chat template
    def format_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return example
        
    print("Applying chat template to dataset...")
    dataset = dataset.map(format_chat_template)
    
    training_args = SFTConfig(
        output_dir="./schgen-qwen-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500, 
        save_steps=100,
        fp16=False,
        bf16=True, 
        optim="adamw_torch",
        dataset_text_field="text", # Moved inside SFTConfig
        max_length=2048,       # Moved inside SFTConfig
    )
    
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args, # SFTTrainer now only needs args and basic components
    )
    
    print("Starting Training!")
    trainer.train()
    
    print("Saving final model...")
    trainer.model.save_pretrained("./schgen-qwen-lora-final")
    tokenizer.save_pretrained("./schgen-qwen-lora-final")
    print("Done!")

if __name__ == "__main__":
    main()
