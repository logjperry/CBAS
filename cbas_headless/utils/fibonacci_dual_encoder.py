import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np 
import h5py
import random
import yaml
import math

import pandas as pd

class Features(Dataset):
    def __init__(self, data_dir, transform=None, bin_size=8, multiple=False):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        self.file_names = os.listdir(data_dir)
        if multiple:
            temp = []
            for fn in self.file_names:
                subdir_path = os.path.join(self.data_dir, fn)

                for f in os.listdir(subdir_path):
                    temp.append(os.path.join(fn, f))

            self.file_names = temp

        self.bin_size = bin_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        subdir_path = os.path.join(self.data_dir, file_name)

        file_name = os.path.split(file_name)[1]

        file_path = os.path.join(subdir_path, file_name+"_outputs.h5")

        with h5py.File(file_path, 'r') as f:
            flow_features = np.array(f['resnet50']['flow_features'][:])
            spatial_features = np.array(f['resnet50']['spatial_features'][:])
            logits = np.array(f['resnet50']['logits'][:])

            features = np.concatenate((spatial_features, flow_features), axis=1)



        # make the indices
        indices = [1,2]

        if self.bin_size-2>=0:
            for i in range(0, self.bin_size-2):
                indices.append(indices[-2]+indices[-1])
        else:
            raise Exception('Bin size must be at least 2.')
        
        reverse = [i*-1 for i in indices[::-1]]
        reverse.extend(indices)


        data = features
        X = []
        # bin the data, at a random value of i
        random_i = random.randint(indices[-1], len(data)-indices[-1]-1)

        X = [data[random_i+j,:] for j in reverse]

        X = np.array(X)

        X = torch.from_numpy(X)

        # for autoencoder, inputs are same as outputs
        return (X, X)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=10):
        super(PositionalEncoding, self).__init__()
        # Create a learnable embedding matrix for positional encodings
        self.positional_encodings = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

    def forward(self, x):
        # x shape is [batch_size, seq_len, d_model]
        # Add positional encoding to each sequence in the batch
        # The broadcasting mechanism will automatically handle the batch dimension
        return x + self.positional_encodings


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


        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_len)

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
        src = self.pos_encoder(src)  # Apply positional encoding

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
        batch_size, seq_len, features = src.shape
        flattened = src.view(batch_size, -1)
        latent = model.encoder_to_latent(flattened)
        
        return latent

    

def train_autoencoder(data_loader, seq_len=16):
    
    input_dim = 1024
    latent_dim = 32
    d_model = 1024
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    lr = 0.0001

    num_epochs = 500

    m_path = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\fibonacci_dual_encoder_32_{seq_len}.pth'
    config_path = os.path.splitext(m_path)[0]+'.yaml'

    model_config = {
        'seq_length':seq_len,
        'input_dim':input_dim,
        'output_dim':input_dim,
        'latent_dim':latent_dim,
        'model_dim':d_model,
        'num_heads':nhead,
        'num_encode_layers':num_encoder_layers,
        'num_decode_layers':num_decoder_layers,
        'learning_rate':lr,
        'model_path':m_path
    }

    with open(config_path, 'w') as file:
        yaml.dump(model_config, file)



    autoencoder = TransformerAutoencoder(input_dim, input_dim, latent_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_len)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
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

            print(f'Epoch {epoch}:')
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
    

    torch.save(autoencoder, m_path)


def smooth(array, half_window):

    result = np.zeros_like(array)

    arr_len = len(array)

    for i in range(0, arr_len):
        array_sum = array[i,:]
        count = 1
        for j in range(-half_window,half_window+1):
            if i+j>=0 and i+j<arr_len and j!=0:
                array_sum = np.add(array_sum,array[i+j,:])
                count+=1

            
        result[i] = array_sum/count

    return result




def load_data(recording_path):
    # Create an instance of your custom dataset

    seq_len = 5


    dataset = Features(data_dir=recording_path, bin_size=seq_len, multiple=True)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    train_autoencoder(data_loader, seq_len=2*seq_len)


def infer(config_path, recording_path):
    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 
                video_loc = os.path.join(subdir, name+'.mp4')
                
                if os.path.exists(video_loc):
                    videos.append(video_loc)
    
    deg_feature_paths = [os.path.splitext(vid)[0]+'_outputs.h5' for vid in videos]

    with open(config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # model_config = {
    #     'seq_length':seq_len,
    #     'input_dim':input_dim,
    #     'output_dim':input_dim,
    #     'latent_dim':latent_dim,
    #     'model_dim':d_model,
    #     'num_heads':nhead,
    #     'num_encode_layers':num_encoder_layers,
    #     'num_decode_layers':num_decoder_layers,
    #     'learning_rate':lr,
    #     'model_path':m_path
    # }

    autoencoder = torch.load(model_config['model_path'])

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    autoencoder.to(device)

    autoencoder.eval()

    for path in deg_feature_paths:

        with h5py.File(path, 'r') as f:
            flow_features = np.array(f['resnet50']['flow_features'][:])
            spatial_features = np.array(f['resnet50']['spatial_features'][:])
            logits = np.array(f['resnet50']['logits'][:])

            features = np.concatenate((spatial_features, flow_features), axis=1)


        # make the indices
        indices = [1,2]
        bin_size = int(model_config['seq_length']/2)

        if bin_size-2>=0:
            for i in range(0, bin_size-2):
                indices.append(indices[-2]+indices[-1])
        else:
            raise Exception('Bin size must be at least 2.')
        
        reverse = [i*-1 for i in indices[::-1]]
        reverse.extend(indices)


        data = features
        X = []
        for i in range(indices[-1], len(data)-indices[-1]-1):
            X.append([data[i+j,:] for j in reverse])
        
        X = np.array(X)

        y = []

        for i in range(0, len(X), 1000):
            start = i 
            end = i+1000

            if end>len(X):
                end = len(X)

            X1 = torch.from_numpy(X[start:end])


            # Move X to the same device as the model
            X1 = X1.to(device)

            with torch.no_grad():
                # Assuming 'input_data' is the data you want to predict on
                # Make sure it's processed as your model expects
                # For example, you might need to add a batch dimension using input_data.unsqueeze(0)

                predictions = extract_latent_space(autoencoder, X1)

                predictions = predictions.cpu()

                predictions = predictions.numpy()

                y.append(predictions)

        total = []

        y = np.concatenate(y, axis=0)

        for i in range(0, indices[-1]):
            total.append(y[0])
        for i in range(indices[-1], len(data)-indices[-1]-1):
            total.append(y[i-indices[-1]])
        for i in range(len(data)-indices[-1]-1, len(data)):
            total.append(y[-1])

        if len(data)!=len(total):
            raise Exception('Lengths do not match!')
        
        print(f'finished with {path}')

        total = np.array(total)

        total = smooth(total, 2)


        csv_path = os.path.splitext(path)[0]+'_'+str(model_config['seq_length'])+'_fibonacci_dual_32_'+'encoded'+'.csv'
        df = pd.DataFrame(total)
        df.to_csv(csv_path)



            