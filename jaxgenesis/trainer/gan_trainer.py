__all__ = ["GANTrainer"]

from .base_trainer import BaseTrainer

## GAN Trainer 

class GANTrainer(BaseTrainer):
    def __init__(self, discriminator_network, generator_network, optimizer, loss_function, epochs):
        super().__init__()
        self.discriminator_network = discriminator_network
        self.generator_network = generator_network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        
    def train(self, data_loader):
        for epoch in range(self.epochs):
            for batch in data_loader:
                ## Train the discriminator Network
                discriminator_data = batch["discriminator_data"]
                discriminator_labels = batch["discriminator_labels"]
                
                ## Train the generator Network
                generator_data = batch["generator_data"]
                generator_labels = batch["generator_labels"]

                ## Configure the Optimizer
                self.optimizer.zero_grad()

                ## Calculate the loss
                discriminator_loss = self.loss_function(self.discriminator_network(discriminator_data), discriminator_labels)
                generator_loss = self.loss_function(self.generator_network(generator_data), generator_labels)

                print("Discriminator Loss: ", discriminator_loss)
                print("Generator Loss: ", generator_loss)
                

                ## Calculate the gradients
                discriminator_loss.backward()
                generator_loss.backward()

                ## Update the weights
                self.optimizer.step()
            


