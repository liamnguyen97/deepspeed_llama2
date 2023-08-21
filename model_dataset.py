import torch
import transformers

class ModelDataset:

    def __init__(self, prompter, tokenizer, max_length):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, prompt):
        result = self.tokenizer(prompt,
                                truncation = True,
                                max_length = self.max_length,
                                padding = True,
                                return_tensors = None)
        
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < self.max_length):            
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    
    def generate_and_tokenize_prompt(self, dataset):
        full_prompt = self.prompter.generate_prompt(dataset["instruction"], dataset["input"], dataset["output"])
        tokenized_prompt = self.tokenize(full_prompt)
        return tokenized_prompt
