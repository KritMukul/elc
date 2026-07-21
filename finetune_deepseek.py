import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def main():
    model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    dataset_path = "master_parallel_dataset.json"

    print("1. Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct"
        # No trust_remote_code, no local_files_only, no legacy flags
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def formatting_prompts_func(example):
        """
        Called per-example (not batched) by newer trl versions.
        Uses the model's OWN chat template (apply_chat_template) rather than a
        hand-rolled format, so training data matches exactly what the model
        was originally instruct-tuned on. Returns a single formatted string.
        """
        graph_str = json.dumps(example['graph_input'])
        spice_str = example['spice_output']

        messages = [
            {
                "role": "user",
                "content": (
                    "You are an expert analog circuit designer. Convert the "
                    "provided netlist JSON graph representation into a valid, "
                    f"simulatable SPICE netlist.\n\n{graph_str}"
                )
            },
            {"role": "assistant", "content": spice_str}
        ]
        # tokenize=False -> we want the raw string, not token ids, since
        # SFTTrainer/formatting_func handles tokenization itself downstream.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return text

    print("2. Loading Model (native BF16 + Flash Attention 2 on H100)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,                 # NOTE: torch_dtype is deprecated, use dtype
        attn_implementation="flash_attention_2",
        device_map="cuda",
        trust_remote_code=True
    )

    print("3. Applying LoRA Adapters...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("4. Loading and Splitting Dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # seed=42 matches the split used for the Qwen run, so results are
    # directly comparable on the exact same held-out test examples.
    dataset_splits = dataset.train_test_split(test_size=0.05, seed=42)

    print("5. Configuring H100 Training Pipeline...")
    training_args = SFTConfig(
        output_dir="./deepseek_circuit_model_output",
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
        max_length=2048,               # renamed from max_seq_length in newer trl
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        # NOTE: no peft_config here -- model is already wrapped via get_peft_model above.
        # Passing it again causes: "You passed a PeftModel instance together with a
        # peft_config to the trainer" error.
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("6. Starting Training Loop...")
    trainer.train()

    print("7. Saving final adapter...")
    trainer.model.save_pretrained("./deepseek_final_circuit_adapter_v2")
    tokenizer.save_pretrained("./deepseek_final_circuit_adapter")
    print("Done.")

if __name__ == "__main__":
    main()