"""Optional QLoRA scaffold. Requires user-selected model, GPU, and large optional dependencies."""
import os
base=os.environ.get('BASE_MODEL')
if not base: raise SystemExit('Set BASE_MODEL to a model you are licensed to use; no model is downloaded by default.')
try:
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments
 from peft import LoraConfig,prepare_model_for_kbit_training,get_peft_model
except ImportError as e: raise SystemExit('Install task09 optional requirements first: '+str(e))
import torch
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(base,quantization_config=quant,device_map='auto'); model=prepare_model_for_kbit_training(model)
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],lora_dropout=.05,task_type='CAUSAL_LM'))
print('Adapter configured:',base,'trainable parameters:',sum(x.numel() for x in model.parameters() if x.requires_grad)); print('Attach a reviewed anonymized dataset and TRL SFTTrainer before training.')
