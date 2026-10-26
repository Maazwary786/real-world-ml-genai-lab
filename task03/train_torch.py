"""Optional PyTorch MLP with batch normalization and dropout."""
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv(Path(__file__).parent / "data/demand.csv")
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns="demand"), data.demand, test_size=0.2, random_state=1)
scaler = StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(x_train.shape[1], 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
for epoch in range(40):
    model.train()
    for features, target in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    mae = torch.mean(torch.abs(model(x_test) - y_test)).item()
print(f"PyTorch MLP test MAE: {mae:.2f}; final training loss: {loss.item():.2f}")
