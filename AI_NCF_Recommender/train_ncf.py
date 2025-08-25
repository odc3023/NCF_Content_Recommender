import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

# Define NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_resources, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.resource_embedding = nn.Embedding(num_resources, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, resource):
        u_emb = self.user_embedding(user)
        r_emb = self.resource_embedding(resource)
        x = torch.cat([u_emb, r_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x

# Training logic wrapped in a main block
if __name__ == "__main__":
    # Load the processed dataset
    df = pd.read_csv("data/processed_kt2.csv")
    print(f"Complete Dataset: {len(df)} records")

    # Convert users and resources to unique indices
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    resource_map = {res: i for i, res in enumerate(df['resource_id'].unique())}

    df['user_id'] = df['user_id'].map(user_map)
    df['resource_id'] = df['resource_id'].map(resource_map)

    # Create a PyTorch dataset
    class EduDataset(Dataset):
        def __init__(self, df):
            self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
            self.resources = torch.tensor(df['resource_id'].values, dtype=torch.long)
            self.labels = torch.tensor(df['interaction'].values, dtype=torch.float)

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            return self.users[idx], self.resources[idx], self.labels[idx]

    # Create Dataset & DataLoader
    dataset = EduDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize Model
    num_users = len(user_map)
    num_resources = len(resource_map)
    model = NCF(num_users, num_resources)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started...")
        epoch_loss = 0
        for batch_idx, (user, resource, label) in enumerate(dataloader):
            optimizer.zero_grad()
            predictions = model(user, resource).squeeze()
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} completed. Loss: {epoch_loss / len(dataloader):.4f}")

    # Save User & Resource Mappings
    with open("user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open("resource_map.pkl", "wb") as f:
        pickle.dump(resource_map, f)

    # Save the trained model
    torch.save(model.state_dict(), "ncf_model.pth")
    print("Model training complete and saved as 'ncf_model.pth'!")