
__all__ = ["VAETrainer"]
from .base_trainer import BaseTrainer


# VAETrainer

class VAETrainer(BaseTrainer):
    """
    VAE Trainer Class
    """
    def __init__(
            self,
            model,
            encoder, 
            decoder, 
            loss_function,
            optimizer, 
            device,
            epochs):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = loss_function
        self.epochs = epochs
        self.optimizer = optimizer
        self.device = device

    def __call__(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.encoder.train()
            self.decoder.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
