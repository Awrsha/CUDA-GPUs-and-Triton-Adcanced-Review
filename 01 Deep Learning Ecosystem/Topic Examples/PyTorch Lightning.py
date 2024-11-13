import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim

# Define the linear regression model within the PyTorch Lightning framework
class LinearModel(pl.LightningModule):
    def __init__(self):
        super(LinearModel, self).__init__()
        # Initialize a single linear layer with 2 input features and 1 output feature
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        # Unpack batch data
        x, y = batch
        # Compute predictions for the batch
        pred = self(x)
        # Calculate Mean Squared Error (MSE) loss between predictions and true labels
        loss = ((pred - y) ** 2).mean()
        return loss

    def configure_optimizers(self):
        # Configure the optimizer for training, using SGD with a learning rate of 0.01
        return optim.SGD(self.parameters(), lr=0.01)

# Prepare data as tensors for input features and target labels
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Input feature tensor
y = torch.tensor([[1.0], [2.0], [3.0]])  # Target label tensor

# Combine features and labels into a dataset
dataset = TensorDataset(x, y)
# Load data in batches for training, with a batch size of 2
dataloader = DataLoader(dataset, batch_size=2)

# Training setup
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10)  # Define training epochs and logging frequency
model = LinearModel()  # Instantiate the linear regression model
trainer.fit(model, dataloader)  # Train the model using the defined trainer and data loader