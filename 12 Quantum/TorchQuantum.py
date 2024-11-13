import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumCircuit(tq.QuantumModule):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Define trainable quantum gates
        self.rx_layers = nn.ModuleList([
            tq.RX(has_params=True, trainable=True)
            for _ in range(n_qubits)
        ])
        
        self.ry_layers = nn.ModuleList([
            tq.RY(has_params=True, trainable=True)
            for _ in range(n_qubits)
        ])
        
        self.rz_layers = nn.ModuleList([
            tq.RZ(has_params=True, trainable=True)
            for _ in range(n_qubits)
        ])
        
        self.crx_layers = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True)
            for _ in range(n_qubits-1)
        ])

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, qdev):
        # Encode classical data into quantum states
        for i in range(self.n_qubits):
            tqf.rx(qdev, wires=i, params=x[:, i])
            
        # Apply trainable quantum gates
        for i in range(self.n_qubits):
            self.rx_layers[i](qdev, wires=i)
            self.ry_layers[i](qdev, wires=i)
            self.rz_layers[i](qdev, wires=i)
            
        # Apply entangling layers
        for i in range(self.n_qubits-1):
            self.crx_layers[i](qdev, wires=[i, i+1])
            qdev.cnot(wires=[i, i+1])
            
        # Measure in computational basis
        return self.measure(qdev)

class HybridModel(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_circuit = QuantumCircuit(n_qubits)
        
        # Classical post-processing layers
        self.fc1 = nn.Linear(n_qubits, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification
        
    def forward(self, x):
        bsz = x.shape[0]
        
        # Initialize quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz)
        
        # Get quantum measurements
        q_out = self.quantum_circuit(x, qdev)
        
        # Classical post-processing
        out = torch.relu(self.fc1(q_out))
        out = self.fc2(out)
        return torch.log_softmax(out, dim=1)

# Training function
def train_model(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.NLLLoss()(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch}: Avg Loss = {total_loss/len(train_loader):.4f}')

# Generate synthetic dataset
def generate_dataset(n_samples=1000, n_qubits=4):
    X = torch.randn(n_samples, n_qubits)
    y = (X.sum(dim=1) > 0).long()
    return X, y

# Main execution
if __name__ == "__main__":
    # Parameters
    N_QUBITS = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 10

    # Generate dataset
    X, y = generate_dataset(n_samples=1000, n_qubits=N_QUBITS)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = HybridModel(n_qubits=N_QUBITS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_model(model, train_loader, optimizer, epochs=EPOCHS)

    # Test prediction
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, N_QUBITS)
        prediction = model(test_input)
        print(f"Test input: {test_input}")
        print(f"Prediction: {torch.exp(prediction)}")