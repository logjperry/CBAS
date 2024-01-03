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
import matplotlib.pyplot as plt

import pandas as pd

class S_Features(Dataset):
    def __init__(self, instance_paths, bin_size=5):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        behaviors = None
        instances = None
        for instance_path in instance_paths:
            with open(instance_path) as file:
                training_set = yaml.safe_load(file)

            if behaviors == None:
                behaviors = training_set['behaviors']

                instances = training_set['instances']
            else:
                for b in behaviors:
                    instances[b].extend(training_set['instances'][b])

        self.behaviors = behaviors 
        self.instances = instances
        self.bin_size = bin_size

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]

        others = [i for i in range(0, len(self.behaviors)) if i != idx]
        random.shuffle(others)

        rand_other_behavior = self.behaviors[others[0]]

        rand_inst1 = random.randint(0,len(self.instances[behavior])-1)
        rand_inst2 = random.randint(0,len(self.instances[behavior])-1)
        rand_other_inst = random.randint(0,len(self.instances[rand_other_behavior])-1)

        data = []
        data_insts = [(behavior, rand_inst1),(behavior, rand_inst2),(rand_other_behavior, rand_other_inst)]
        
        for b, inst_num in data_insts:
            
            inst = self.instances[b][inst_num]

            video = inst['video']
            start = inst['start']
            end = inst['end']

            file_path = os.path.splitext(video)[0]+'_outputs.h5'

            with h5py.File(file_path, 'r') as f:
                flow_features = np.array(f['resnet50']['flow_features'][:])
                spatial_features = np.array(f['resnet50']['spatial_features'][:])

                features = np.concatenate((spatial_features, flow_features), axis=1)

            feats = features

            # make the indices
            indices = [1,2]

            if self.bin_size-2>=0:
                for i in range(0, self.bin_size-2):
                    indices.append(indices[-2]+indices[-1])
            else:
                raise Exception('Bin size must be at least 2.')
            
            reverse = [i*-1 for i in indices[::-1]]
            reverse.extend(indices)

            random_i = 0

            if start >= indices[-1] and end < len(feats)-indices[-1]:
                random_i = random.randint(start, end-1)
            else:
                if start<indices[-1]:
                    start = indices[-1]
                if end>=len(feats)-indices[-1]:
                    end = len(feats)-indices[-1]

                if end-1<start:
                    random_i = random.randint(indices[-1], len(feats)-indices[-1]-1)
                else:
                    random_i = random.randint(start, end-1)


            X = []

            X = [feats[random_i+j,:] for j in reverse]

            X = np.array(X)

            X = torch.from_numpy(X)

            data.append(X)

        # return original encoding, similar encoding, contrastive encoding
        return (data[0], data[1], data[2])

class US_Features(Dataset):
    def __init__(self, data_dir, transform=None, bin_size=5, multiple=False):
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

        XT = np.transpose(X)

        X = torch.from_numpy(X)
        XT = torch.from_numpy(XT)


        # for autoencoder, inputs are same as outputs
        return (X, XT)
    

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

    def forward(self, tgt):
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

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model, nhead, num_encoder_layers, seq_len):
        super(TransformerEncoder, self).__init__()

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_len)

        self.encoder_embedding = nn.Linear(input_dim, d_model)
        
        # Flatten the last two dimensions
        self.flatten = nn.Flatten(start_dim=1)  

        # Compress to latent space
        self.encoder_to_latent = nn.Linear(seq_len * d_model, latent_dim)  
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)])
        

    def forward(self, src):
        src = self.encoder_embedding(src)
        src = self.pos_encoder(src)  # Apply positional encoding

        for layer in self.encoder_layers:
            src = layer(src)
        
        # Compress to latent space
        flattened = self.flatten(src)
        latent = self.encoder_to_latent(flattened)

        return latent
    
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, d_model, nhead, num_decoder_layers, seq_len):
        super(TransformerDecoder, self).__init__()

        # Expand from latent space
        self.latent_to_decoder = nn.Linear(latent_dim, 1024 * d_model)  

        # Unflatten to original shape
        self.unflatten = nn.Unflatten(1, (1024, d_model))  
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, latent):

        # Expand from latent space
        expanded = self.latent_to_decoder(latent)
        tgt = self.unflatten(expanded)

        for layer in self.decoder_layers:
            tgt = layer(tgt)

        output = self.output_layer(tgt)
        return output
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, src):

        latent = self.encoder(src)
        output = self.decoder(latent)

        return output

def train_autoencoder(us_data_loader, s_data_loader, seq_len=16):
    
    input_dim = 1024
    latent_dim = 32
    d_model = 1024
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    lr = 0.0001

    num_epochs = 1000

    m_path = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\transpose_encoder.pth'
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



    encoder = TransformerEncoder(input_dim, latent_dim, d_model, nhead, num_encoder_layers, seq_len)
    decoder = TransformerDecoder(latent_dim, 10, d_model, nhead, num_decoder_layers, seq_len)
    autoencoder = Autoencoder(encoder, decoder)

    optimizer_total = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    criterion_total = nn.MSELoss()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    autoencoder.to(device)

    EL = []
    AL = []


    # Training loop
    for epoch in range(num_epochs):

        
        for inputs, targets in us_data_loader:

            print(f'Epoch {epoch}:')
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass, backward pass, and optimize
            optimizer_total.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion_total(outputs, targets)
            
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
            optimizer_total.step()

            del inputs  # Delete the GPU tensor
            del outputs  # Delete the GPU tensor
            del targets  # Delete the GPU tensor
        
        if (epoch+1)%20==0:
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




def load_data(recording_path, instance_paths):

    # -8 -5 -3 -2 -1 1 2 3 5 8
    seq_len = 5

    s_dataset = S_Features(instance_paths=instance_paths, bin_size=seq_len)
    us_dataset = US_Features(data_dir=recording_path, bin_size=seq_len, multiple=True)

    # Create a DataLoader
    s_data_loader = DataLoader(s_dataset, batch_size=1024, shuffle=True, num_workers=8)
    us_data_loader = DataLoader(us_dataset, batch_size=512, shuffle=True, num_workers=8)

    train_autoencoder(us_data_loader, s_data_loader, seq_len=2*seq_len)


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

                predictions = autoencoder.encoder(X1)

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


        csv_path = os.path.splitext(path)[0]+'_transpose_'+'encoded'+'.csv'
        df = pd.DataFrame(total)
        df.to_csv(csv_path)



            