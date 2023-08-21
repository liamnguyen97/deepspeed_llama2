import torch
from transformers import AutoTokenizer, AutoModelForCausalM
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

class ModelConfig:

    def tokenizer(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenizer.pad_token = tokenizer.eos_toke
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    def load_pretrained_model(self, model_checkpoint, device_map):
        model = AutoModelForCausalM.from_pretrained(model_checkpoint)
        return model
    
    def add_lora(self, model, r: int, lora_alpha: int, lora_dropout: float):
        lora_config = LoraConfig(r = r,
                                 lora_alpha = lora_alpha,
                                 lora_dropout = lora_dropout,
                                 bias = "none",
                                 task_type = "CAUSAL_LM")
        lora_model = get_peft_model(model, lora_config)
        return lora_model
    
    