"""Optional 4-bit QLoRA fine-tuning for a user-selected, licensed base model."""
import json
import os
from pathlib import Path

base_model = os.environ.get("BASE_MODEL")
if not base_model:
    raise SystemExit("Set BASE_MODEL to a model you are licensed to use; model weights are never downloaded automatically.")

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    raise SystemExit("Install task09/requirements-optional.txt before training: " + str(exc)) from exc

rows = [json.loads(line) for line in (Path(__file__).parent / "data/claims.jsonl").read_text().splitlines()]
examples = [{"text": f"System: {row['system']}\nUser: {row['user']}\nAssistant: {row['assistant']}"} for row in rows]
dataset = Dataset.from_list(examples)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quantization, device_map="auto")
adapter = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
training = SFTConfig(
    output_dir="outputs/claims-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    max_length=512,
    report_to=[],
)
trainer = SFTTrainer(model=model, args=training, train_dataset=dataset, processing_class=tokenizer, peft_config=adapter)
result = trainer.train()
trainer.save_model(training.output_dir)
print({"train_loss": result.training_loss, "global_steps": result.global_step, "adapter_dir": training.output_dir})
