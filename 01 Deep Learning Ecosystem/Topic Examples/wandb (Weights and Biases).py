import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize wandb
wandb.init(project='pytorch-example')

# Define a simple model
class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Log model and hyperparameters
wandb.config.lr = 0.01
wandb.config.epochs = 10

# Training loop with wandb logging
for epoch in range(wandb.config.epochs):
    # Dummy data and loss
    input_data = torch.randn(10)
    target = input_data * 2
    output = model(input_data)
    loss = criterion(output, target)
    
    # Log loss value to wandb
    wandb.log({"epoch": epoch, "loss": loss.item()})

# Finish the experiment
wandb.finish()