from model_config import ModelConfig
from model_prompt import Prompter
from model_dataset import ModelDataset
from deepspeed_trainer import DeepspeedTrainer
import torch
import deepspeed
import os
import transformers
import datasets 
from datasets import load_dataset
if __name__ == "__main__":
    
    model_config = ModelConfig()
    tokenizer = model_config.tokenizer(model_checkpoint="meta-llama/Llama-2-7b-hf")
    model = model_config.load_pretrained_model(model_checkpoint="meta-llama/Llama-2-7b-hf")

    prompter = Prompter()
    dataset = load_dataset("MBZUAI/Bactrian-X","vi",split="train")

    # splitted_dataset = dataset.train_test_split(test_size=0.1, seed = 42)
    model_inputs = ModelDataset(prompter=prompter, tokenizer=tokenizer, max_length=512)
    train_data = dataset["train"].shuffle().map(model_inputs.generate_and_tokenize_prompt)
    train_data.set_format("torch")

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    train_batch_size = 4 * world_size

    ds_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 100,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 3e-5,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },

            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 3e-5,
                    "warmup_num_steps": 500
                }
            },

            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 1e6,
                "stage3_prefetch_bucket_size": 0.94e6,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 3e7,
                "stage3_prefetch_bucket_size": 3e7,
                "memory_efficient_linear": False
            },
            "steps_per_print": 300,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,
            "wall_clock_breakdown": False
        }
    
    ds_engine, optimizer, train_dataloader, _ = deepspeed.initialize(model=model,training_data=train_data, 
                                                collate_fn = transformers.DataCollatorForSeq2Seq(tokenizer = tokenizer,
                                                padding = True,return_tensors = "pt"),  config_params=ds_config)

    trainer = DeepspeedTrainer(lr = 1e-4,
                      epochs = 4,
                      model = ds_engine,                  
                      optimizer = optimizer)
    
    trainer.train(train_dataloader = train_dataloader,display_steps = 500,save_steps=1000)
    
    
