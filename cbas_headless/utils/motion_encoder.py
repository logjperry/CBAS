import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np 
import h5py
import random

class Features(Dataset):
    def __init__(self, data_dir, transform=None, bin_size=5):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = os.listdir(data_dir)
        self.bin_size = bin_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        subdir_path = os.path.join(self.data_dir, file_name)

        deg_file_path = os.path.join(subdir_path, file_name+"_outputs.h5")

        dlc_file_path = os.path.join(subdir_path, )

        with h5py.File(deg_file_path, 'r') as f:
            flow_features = np.array(f['resnet50']['flow_features'][:])
            spatial_features = np.array(f['resnet50']['spatial_features'][:])
            logits = np.array(f['resnet50']['logits'][:])

        # open DLC features
        with h5py.File(dlc_file_path, 'r') as file:
            def print_name(name):
                item = file[name]
                if isinstance(item, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {item.shape}, Dtype: {item.dtype}")
                else:
                    print(f"Group: {name}")

            # Recursively visit all groups and datasets in the file
            file.visit(print_name)


        # make the bin size odd
        adj_binsize = self.bin_size
        if self.bin_size%2==0:
            adj_binsize+=1
        adj_binsize = int(adj_binsize/2)

        deg_data = logits
        

        # make sure that the lengths are the same, get in the format of (1, F)

        X = []
        y = []
        # bin the data, at a random value of i
        random_i = random.randint(self.bin_size, len(deg_data)-self.bin_size)

        X = np.array(deg_data[(random_i-adj_binsize):(random_i+adj_binsize+1),:])

        X = torch.from_numpy(X)

        # for autoencoder, inputs are same as outputs
        return (X, X)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2)[0])
        src2 = self.norm2(src)
        # Apply the first linear layer and ReLU activation
        src2 = F.relu(self.linear1(src2))
        src2 = self.dropout2(src2)
        # Apply the second linear layer
        src2 = self.linear2(src2)  # Ensure this layer outputs a tensor with the last dimension 512

        # Add the output of feedforward network to the original input (residual connection)
        src = src + self.dropout(src2)
        # src = src + self.dropout(self.linear2(src))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(tgt2, tgt2, tgt2)[0])
        tgt2 = self.norm2(tgt)
        
        # Feedforward network
        tgt2 = F.relu(self.linear1(tgt2))
        tgt2 = self.dropout3(tgt2)
        tgt2 = self.linear2(tgt2)  # Ensure this layer outputs a tensor with the last dimension 512
        
        # Add the output of feedforward network to the original input (residual connection)
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_len):
        super(TransformerAutoencoder, self).__init__()
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        
        # Flatten the last two dimensions
        self.flatten = nn.Flatten(start_dim=1)  

        # Compress to latent space
        self.encoder_to_latent = nn.Linear(seq_len * d_model, latent_dim)  

        # Expand from latent space
        self.latent_to_decoder = nn.Linear(latent_dim, seq_len * d_model)  

        # Unflatten to original shape
        self.unflatten = nn.Unflatten(1, (seq_len, d_model))  
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.encoder_embedding(src)
        for layer in self.encoder_layers:
            src = layer(src)
        
        # Compress to latent space
        flattened = self.flatten(src)
        latent = self.encoder_to_latent(flattened)

        # Expand from latent space
        expanded = self.latent_to_decoder(latent)
        tgt = self.unflatten(expanded)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src)

        output = self.output_layer(tgt)
        return output
    
def extract_latent_space(model, src):
    with torch.no_grad():
        src = model.encoder_embedding(src)
        for layer in model.encoder_layers:
            src = layer(src)
        
        # Compress to latent space
        flattened = model.flatten(src)
        latent = model.encoder_to_latent(flattened)
        return latent

    

def train_autoencoder(data_loader, seq_len=5):
    
    input_dim = 9
    latent_dim = 16
    d_model = 512
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4

    num_epochs = 500

    autoencoder = TransformerAutoencoder(input_dim, latent_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_len)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    autoencoder.to(device)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass, backward pass, and optimize
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)
            
            outputs = outputs.cpu()
            targets = targets.cpu()

            # Flatten the tensors
            outputs_flat = outputs.view(outputs.size(0), -1)  # Reshapes to (batch_size, seq_len*9)
            targets_flat = targets.view(targets.size(0), -1)  # Reshapes to (batch_size, seq_len*9)

            # Compute MSE Loss
            mse_loss = F.mse_loss(outputs_flat, targets_flat, reduction='mean')

            # Compute Variance of y
            var_y = targets_flat.var(dim=0)

            # Compute R^2 Score
            r_squared = 1 - mse_loss / var_y.mean()

            print("R^2 Score:", r_squared.item())

            loss.backward()
            optimizer.step()

            del inputs  # Delete the GPU tensor
            del outputs  # Delete the GPU tensor
            torch.cuda.empty_cache()  # Clear GPU cache
    

    torch.save(autoencoder, f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\motion_encoder_{seq_len}_frame.pth')



def load_data(recording_path):
    # Create an instance of your custom dataset

    # seq_len must be odd
    seq_len = 5

    if seq_len%2==0:
        seq_len+=1

    dataset = Features(data_dir=recording_path, bin_size=seq_len)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4)

    train_autoencoder(data_loader, seq_len=seq_len)