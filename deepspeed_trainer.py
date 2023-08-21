import torch
from tqdm.auto import tqdm

class DeepspeedTrainer:

    def __init__(self, lr: float, epochs: int, model, optimizer):
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer

    def train(self, train_dataloader, display_steps, save_steps):
        num_update_steps_per_epoch = len(train_dataloader)
        total_loss = 0
        num_steps = num_update_steps_per_epoch * self.epochs
        progress_bar = tqdm(range(num_steps))
        current_steps = 0
        self.model.train()
        for epoch in range(self.epochs):
            for batch in train_dataloader:
                batch = {k:v.to(self.model.device) for k,v in batch.items()}
                outputs = self.model()

                loss = outputs.loss
                total_loss += loss.item()
                self.model.backward(loss)
                self.model.step()

                progress_bar.update(1)
                current_steps += 1
                if current_steps % display_steps == 0:
                    print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss / current_steps}')
                if current_steps % save_steps == 0:
                    print('Saving 16bit weight model')
                    print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss / current_steps}')
                    self.model.save_16bitmodel("output/latest")