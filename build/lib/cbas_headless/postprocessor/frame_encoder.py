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


class Frames(Dataset):
    def __init__(self, data_dir, video_length, transform=None, multiple=False):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.video_length = video_length

        self.file_names = os.listdir(data_dir)
        if multiple:
            temp = []
            for fn in self.file_names:
                subdir_path = os.path.join(self.data_dir, fn)

                for f in os.listdir(subdir_path):
                    temp.append(os.path.join(fn, f))

            self.file_names = temp

        self.features = {vid:[] for vid in self.file_names}


    def __len__(self):
        return len(self.file_names)*100

    def __getitem__(self, idx):
        fidx = int(idx/100)
        file_name = self.file_names[fidx]

        subdir_path = os.path.join(self.data_dir, file_name)
        fn = os.path.split(file_name)[1]

        file_path = os.path.join(subdir_path, fn+"_outputs.h5")
        random_i = random.randint(0, self.video_length)

        try:

            with h5py.File(file_path, 'r') as f:
                try:
                    flow_features = np.array(f['resnet50']['flow_features'][random_i])
                    spatial_features = np.array(f['resnet50']['spatial_features'][random_i])
                except:
                    length = len(np.array(f['resnet50']['flow_features'][:]))
                    random_i = random.randint(0, length-1)
                    flow_features = np.array(f['resnet50']['flow_features'][random_i])
                    spatial_features = np.array(f['resnet50']['spatial_features'][random_i])

                features = np.concatenate((spatial_features, flow_features), axis=0)
        except:
            print(f'Error opening {file_path}')
            return self.__getitem__(idx+1)



        data = features

        X = data

        X = np.array(X)
        XP = np.copy(X)
        
        ablations = random.sample(range(len(X)), 256)
        var = np.var(X)
        additions = [random.random()*var**2 - (var**2)/2 for i in range(len(ablations))]
        XP[ablations] += additions

        ablations = np.array(ablations)


        return (XP, X, ablations)
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        # Create a learnable embedding matrix for positional encodings
        self.positional_encodings = nn.Parameter(torch.randn(d_model) * 0.01)

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

    def forward(self, input, latent):
        input1 = self.norm1(input)
        input = input + self.dropout1(self.self_attn(input1, input1, input1)[0])
        input1 = self.norm2(input)

        input1 = input1 + self.dropout1(self.multihead_attn(input1, latent, latent)[0])
        input2 = self.norm2(input1)
        
        # Feedforward network
        input2 = F.relu(self.linear1(input2))
        input2 = self.dropout3(input2)
        input2 = self.linear2(input2)  # Ensure this layer outputs a tensor with the last dimension 512
        
        # Add the output of feedforward network to the original input (residual connection)
        output = input + self.dropout2(input2)
        return output

class FrameEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model, nhead, num_encoder_layers):
        super(FrameEncoder, self).__init__()

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_embedding = nn.Linear(input_dim, d_model)

        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)])

        
        self.dropout = nn.Dropout(0.2)

        # Compress to latent space
        self.encoder_to_latent = nn.Linear(d_model, latent_dim)   
        

    def forward(self, src):
        src = self.encoder_embedding(src)
        src = self.pos_encoder(src)  # Apply positional encoding

        for layer in self.encoder_layers:
            src = layer(src)

        src = self.encoder_to_latent(src)

        return src
    
class FrameDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, d_model, nhead, num_decoder_layers, dropout=0.2):
        super(FrameDecoder, self).__init__()

        # Expand from latent space
        self.latent_to_decoder = nn.Linear(latent_dim, d_model)  
        self.input_embedding = nn.Linear(input_dim, d_model)  
        
        self.pos_encoder1 = PositionalEncoding(d_model)
        self.pos_encoder2 = PositionalEncoding(d_model)

        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)])
        
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, input, latent):

        input = self.input_embedding(input)
        input = self.pos_encoder1(input)

        latent = self.latent_to_decoder(latent)
        latent = self.pos_encoder2(latent)

        for layer in self.decoder_layers:
            input = layer(input, latent)

        output = self.output_layer(input)

        return output
    
    def encode(self, input, latent):
        input = self.input_embedding(input)
        input = self.pos_encoder1(input)

        latent = self.latent_to_decoder(latent)
        latent = self.pos_encoder2(latent)

        for layer in self.decoder_layers:
            input = layer(input, latent)
    
        return input

 
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, src):

        latent = self.encoder(src)
        output = self.decoder(src, latent)

        return output

    def encode(self, src):
        latent = self.encoder(src)
        output = self.decoder.encode(src, latent)

        return output




def train_autoencoder(us_data_loader, model_path, latent_dim = 64, d_model = 128, nhead = 64, num_encoder_layers = 15, num_decoder_layers = 15, lr = 0.00001, num_epochs = 5000):
    

    m_path = model_path
    config_path = os.path.splitext(m_path)[0]+'.yaml'

    model_config = {
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



    encoder = FrameEncoder(input_dim, latent_dim, d_model, nhead, num_encoder_layers)
    decoder = FrameDecoder(input_dim, latent_dim, input_dim, d_model, nhead, num_decoder_layers, dropout=0.2)
    autoencoder = Autoencoder(encoder, decoder)
    
    optimizer_total = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    criterion_total = nn.MSELoss()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    autoencoder.to(device)



    # Training loop
    for epoch in range(num_epochs):

        
        for inputs, targets, ablations in us_data_loader:


            print(f'Epoch {epoch}:')
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass, backward pass, and optimize
            optimizer_total.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion_total(outputs, targets)


            loss.backward()
            optimizer_total.step()


            outputs = outputs.cpu()
            targets = targets.cpu()
            # Flatten the tensors
            outputs_flat = outputs  # Reshapes to (batch_size, seq_len*9)
            targets_flat = targets  # Reshapes to (batch_size, seq_len*9)
            # Compute MSE Loss
            mse_loss = F.mse_loss(outputs_flat, targets_flat, reduction='mean')
            # Compute Variance of y
            var_y = targets_flat.var(dim=1)
            # Compute R^2 Score
            r_squared = 1 - mse_loss / var_y.mean()
            print("R^2 Score:", r_squared.item())


            batch_size, num_features = outputs.shape
            ablation_mask = torch.zeros(batch_size, num_features, dtype=torch.bool)

            for i in range(batch_size):
                ablation_mask[i, ablations[i]] = True

            outputs_ablated = torch.masked_select(outputs, ablation_mask)
            targets_ablated = torch.masked_select(targets, ablation_mask)

            mse_ablated = torch.mean((outputs_ablated - targets_ablated) ** 2)

            print("MSE for all values:", mse_loss.item())
            print("MSE for ablated values:", mse_ablated.item())

        
        if (epoch+1)%20==0:
            torch.save(autoencoder, m_path)


def load_postprocessors(project_config='undefined'):
    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the models
        models = pconfig['models_path']
        # extract the mdoels config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return mconfig
    else:
        if os.path.exists(project_config):
            print('Project found.')
        else:
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the models
        models = pconfig['models_path']
        # extract the mdoels config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return mconfig



def train_on_data(recording_path, model_name, video_length=18000):


    us_dataset = Frames(data_dir=recording_path, video_length=video_length, multiple=True)

    # Create a DataLoader
    us_data_loader = DataLoader(us_dataset, batch_size=512, shuffle=True, num_workers=8)





    train_autoencoder(us_data_loader, model_path)




def infer(config_path, recording_path, videos):

    
    
    
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

            features = np.concatenate((spatial_features, flow_features), axis=1)



        data = features
        X = features
        
        X = np.array(X)

        y = []

        for i in range(0, len(X), 1000):
            start = i 
            end = i+1000

            if end>len(X):
                end = len(X)

            X1 = torch.from_numpy(X[start:end])
            X1 = X1.unsqueeze(0)


            # Move X to the same device as the model
            X1 = X1.to(device)

            with torch.no_grad():

                predictions = autoencoder.encode(X1)

                predictions = predictions.cpu()

                predictions = predictions.numpy()

                y.extend(predictions[0,:])


        total = y

        if len(data)!=len(total):
            raise Exception('Lengths do not match!')
        

        total = np.array(total)

        # Specify the filename for the HDF5 file
        filename = os.path.splitext(path)[0]+'_encoded_features.h5'

        # Open the HDF5 file in write mode
        with h5py.File(filename, "w") as f:
            # Create a dataset to store the matrix
            dset = f.create_dataset("features", data=total)
        
        print(f'finished with {path}')



            