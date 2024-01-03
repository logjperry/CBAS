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
from scipy.stats import pearsonr

import pandas as pd

class S_Features(Dataset):
    def __init__(self, instance_paths, behavior):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        
        self.indices = [0,4]

        instances = []
        for instance_path in instance_paths:
            with open(instance_path) as file:
                training_set = yaml.safe_load(file)

                for inst in training_set['instances'][behavior]:
                    file_path = os.path.splitext(inst['video'])[0]+'_outputs.h5'

                    with h5py.File(file_path, 'r') as f:
                        logits = np.array(f['resnet50']['logits'][:])

                    if inst['start']>self.indices[-1] and inst['end']<len(logits)-self.indices[-1]-2:
                        instances.append(inst)

        self.behavior = behavior 
        self.instances = instances

        self.features = {inst['video']:None for inst in self.instances}

    def __len__(self):
        return len(self.instances)*10

    def __getitem__(self, idx):
        iidx = int(idx/10)
        inst = self.instances[iidx]


        video = inst['video']
        start = inst['start']
        end = inst['end']

        if self.features[video] is None:

            file_path = os.path.splitext(video)[0]+'_outputs.h5'

            file_path = os.path.splitext(file_path)[0]+"_encoded.h5"

            with h5py.File(file_path, 'r') as f:
                features = np.array(f['features'])
            
            self.features[video] = features

        data = self.features[video]

        
        reverse = [i*-1 for i in self.indices[::-1] if i != 0]
        reverse.extend(self.indices)

        random_i = 0

        if start >= self.indices[-1]+1 and end < len(data)-self.indices[-1]-2:
            random_i = random.randint(start, end-1)

        X = []

        X = [data[random_i+j,:] for j in reverse]
        X = np.array(X)
        Y = [data[random_i+j,:] for j in range(reverse[0],reverse[-1]+1)]
        Y = np.array(Y)

        X_prime = np.zeros_like(Y)

        for r in range(len(X_prime[0,:])):
            row = np.interp([-4,-3,-2,-1,0,1,2,3,4], [-4,0,4], X[:,r])

            X_prime[:,r] = row


        X_prime = torch.from_numpy(X_prime)
        Y = torch.from_numpy(Y)
        

        # return original encoding, similar encoding, contrastive encoding
        return (X_prime, Y)

class US_Features(Dataset):
    def __init__(self, data_dir, transform=None, bin_size=5, multiple=False):
        """
        Args:
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        self.indices = [0,4]

        self.file_names = os.listdir(data_dir)
        if multiple:
            temp = []
            for fn in self.file_names:
                subdir_path = os.path.join(self.data_dir, fn)

                for f in os.listdir(subdir_path):
                    temp.append(os.path.join(fn, f))

            self.file_names = temp

        self.bin_size = bin_size

        self.features = {vid:[] for vid in self.file_names}

    def __len__(self):
        return len(self.file_names)*100

    def __getitem__(self, idx):
        fidx = int(idx/100)
        file_name = self.file_names[fidx]
        subdir_path = os.path.join(self.data_dir, file_name)

        if len(self.features[file_name])==0:

            file_name = os.path.split(file_name)[1]

            file_path = os.path.join(subdir_path, file_name+"_outputs_encoded.h5")

            with h5py.File(file_path, 'r') as f:
                features = np.array(f['features'])

            self.features[file_name] = features


        
        reverse = [i*-1 for i in self.indices[::-1] if i != 0]
        reverse.extend(self.indices)


        data = self.features[file_name]
        X = []
        # bin the data, at a random value of i
        random_i = random.randint(self.indices[-1], len(data)-self.indices[-1]-1)

        X = [data[random_i+j,:] for j in reverse]
        X = np.array(X)
        Y = [data[random_i+j,:] for j in range(reverse[0],reverse[-1]+1)]
        Y = np.array(Y)

        X_prime = np.zeros_like(Y)

        for r in range(len(X_prime[0,:])):
            row = np.interp([-4,-3,-2,-1,0,1,2,3,4], [-4,0,4], X[:,r])

            X_prime[:,r] = row



        X_prime = torch.from_numpy(X_prime)

        # for autoencoder, inputs are same as outputs
        return (X_prime, Y)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len=9):
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
        self.latent_to_decoder = nn.Linear(latent_dim, seq_len * d_model)  

        # Unflatten to original shape
        self.unflatten = nn.Unflatten(1, (seq_len, d_model))  
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
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, d_model, nhead, num_decoder_layers, seq_len):
        super(Decoder, self).__init__()

        # Expand from latent space
        self.latent_to_decoder = nn.Linear(latent_dim, seq_len * d_model)  

        # Unflatten to original shape
        self.unflatten = nn.Unflatten(1, (seq_len, d_model))  
        self.decoder_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_decoder_layers)])
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
    
def calculate_mase(actual, predicted, seasonal_period=1):
    n = len(actual)
    numerator = np.mean(np.abs(actual - predicted))
    
    # Calculate the denominator (mean absolute error of the naive forecast)
    naive_forecast = np.roll(actual, seasonal_period)
    naive_forecast[:seasonal_period] = np.nan  # Set NaN for the initial values
    denominator = np.mean(np.abs(actual - naive_forecast))
    
    return numerator / denominator

def calculate_mape(actual, predicted):

    actual[actual==0] += .001

    return np.mean(np.abs((actual - predicted) / actual)) * 100

def train_autoencoder(us_data_loader, seq_len=16):
    
    input_dim = 64
    latent_dim = 32
    d_model = 1024
    nhead = 16
    num_encoder_layers = 6
    num_decoder_layers = 6
    lr = 0.0001

    num_epochs = 1000

    m_path = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\interpolative.pth'
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
    decoder = TransformerDecoder(latent_dim, input_dim, d_model, nhead, num_decoder_layers, seq_len)
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
            outputs_flat = outputs.view(outputs.size(0), -1)  # Reshapes to (batch_size, seq_len*64)
            targets_flat = targets.view(targets.size(0), -1)  # Reshapes to (batch_size, seq_len*64)

            # Convert tensors to NumPy arrays for Pearson's correlation calculation
            model_outputs_np = outputs_flat.cpu().detach().numpy()
            target_values_np = targets_flat.cpu().detach().numpy()

            # Calculate Pearson's correlation for each sample in the batch
            both = [(mo, tv) for mo,tv in zip(model_outputs_np,target_values_np)]
            correlation_coefficients = np.array([pearsonr(output, target)[0] for output, target in both])
            mean_mape = np.array([calculate_mape(output, target) for output, target in both])

            max_mape = np.argmax(mean_mape)

            output, target = both[max_mape]
            plt.clf()
            for i in range(0, len(output), 64):

                plt.plot(np.ones(64)*i/64,target[i:i+64]+np.arange(0,64)*.1,'go')
                plt.plot(np.ones(64)*i/64,output[i:i+64]+np.arange(0,64)*.1,'r+')
            
            plt.savefig(os.path.splitext(m_path)[0]+"_max_MAPE.png")

            # Calculate the mean correlation coefficient across the batch
            mean_correlation = np.mean(correlation_coefficients)

            print("Mean Pearson's Correlation Coefficient:", mean_correlation)
            print("MAPE:", np.mean(mean_mape))
            AL.append((epoch, mean_mape))

            loss.backward()
            optimizer_total.step()
        
        # plt.clf()

        # eps = [x for x,y in EL]
        # cll = [y for x,y in EL]

        # plt.plot(eps,cll)

        # eps = [x for x,y in AL]
        # ell = [y for x,y in AL]

        # plt.plot(eps,ell)

        # plt.savefig(os.path.splitext(m_path)[0]+".png")

        if (epoch+1)%50==0:
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

    us_dataset = US_Features(data_dir=recording_path, multiple=True)

    # Create a DataLoader
    us_data_loader = DataLoader(us_dataset, batch_size=1024, shuffle=True, num_workers=8)

    train_autoencoder(us_data_loader, seq_len=9)


def make_specialists(config_path, instance_paths, behaviors):
    for behavior in behaviors:

        s_dataset = S_Features(instance_paths=instance_paths, behavior=behavior)
        s_data_loader = DataLoader(s_dataset, batch_size=1024, shuffle=True, num_workers=8)

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

        
        optimizer_total = torch.optim.Adam(autoencoder.parameters(), lr=model_config['learning_rate']*.1)

        criterion_total = nn.MSELoss()


        num_epochs = 500

        m_path = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\specialists\\interpolative_'+behavior+'.pth'
        config_path = os.path.splitext(m_path)[0]+'.yaml'

        
        model_config['model_path'] = m_path

        with open(config_path, 'w') as file:
            yaml.dump(model_config, file)


        AL = []


        # Training loop
        for epoch in range(num_epochs):

            for inputs, targets in s_data_loader:

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
                AL.append((epoch, r_squared.item()))

                # Convert tensors to NumPy arrays for Pearson's correlation calculation
                model_outputs_np = outputs_flat.cpu().detach().numpy()
                target_values_np = targets_flat.cpu().detach().numpy()

                # Calculate Pearson's correlation for each sample in the batch
                both = [(mo, tv) for mo,tv in zip(model_outputs_np,target_values_np)]
                correlation_coefficients = np.array([pearsonr(output, target)[0] for output, target in both])
                mean_mape = np.array([calculate_mape(output, target) for output, target in both])

                max_mape = np.argmax(mean_mape)

                output, target = both[max_mape]
                plt.clf()
                for i in range(0, len(output), 64):

                    plt.plot(np.ones(64)*i/64,target[i:i+64]+np.arange(0,64)*.1,'go')
                    plt.plot(np.ones(64)*i/64,output[i:i+64]+np.arange(0,64)*.1,'r+')
                
                plt.savefig(os.path.splitext(m_path)[0]+"_max_MAPE.png")

                # Calculate the mean correlation coefficient across the batch
                mean_correlation = np.mean(correlation_coefficients)

                print("Mean Pearson's Correlation Coefficient:", mean_correlation)
                print("MAPE:", np.mean(mean_mape))

                loss.backward()
                optimizer_total.step()

            
            plt.clf()

            eps = [x for x,y in AL]
            ell = [y for x,y in AL]

            plt.plot(eps,ell)

            plt.savefig(os.path.splitext(m_path)[0]+".png")

            if (epoch+1)%50==0:
                torch.save(autoencoder, m_path)


def infer_loss(config_path, recording_path, name):
    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for n in dirs:
                subdir = os.path.join(root, n) 
                video_loc = os.path.join(subdir, n+'.mp4')
                
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
        indices = [0,4]
        
        reverse = [i*-1 for i in indices[::-1] if i != 0]
        reverse.extend(indices)


        data = features
        inputs = []
        targets = []
        for i in range(reverse[-1], len(data)-reverse[-1]-1):
            X = [data[i+j,:] for j in reverse]
            X = np.array(X)

            Y = [data[i+j,:] for j in range(reverse[0],reverse[-1]+1)]
            Y = np.array(Y)

            X_prime = np.zeros_like(Y)

            for r in range(len(X_prime[0,:])):
                row = np.interp([-4,-3,-2,-1,0,1,2,3,4], [-4,0,4], X[:,r])

                X_prime[:,r] = row
            
            # X_prime = torch.from_numpy(X_prime)
            # Y = torch.from_numpy(Y)

            inputs.append(X_prime)
            targets.append(Y)


        loss = []

        for i in range(0, len(inputs), 1000):
            start = i 
            end = i+1000

            if end>len(inputs):
                end = len(inputs)

            X1 = torch.from_numpy(np.array(inputs[start:end]))
            print(X1.shape)

            Y1 = np.array(targets[start:end])


            # Move X to the same device as the model
            X1 = X1.to(device)

            with torch.no_grad():
                # Assuming 'input_data' is the data you want to predict on
                # Make sure it's processed as your model expects
                # For example, you might need to add a batch dimension using input_data.unsqueeze(0)

                predictions = autoencoder(X1)

                predictions = predictions.cpu()

                predictions = predictions.numpy()

            for p in range(0,len(predictions)):
                mse = (np.square(Y1[p] - predictions[p])).mean()
                loss.append(mse)

        total = []


        for i in range(0, indices[-1]):
            total.append(loss[0])
        for i in range(indices[-1], len(data)-indices[-1]-1):
            total.append(loss[i-indices[-1]])
        for i in range(len(data)-indices[-1]-1, len(data)):
            total.append(loss[-1])

        if len(data)!=len(total):
            raise Exception('Lengths do not match!')
        
        print(f'finished with {path}')

        total = np.array(total)

        #total = smooth(total, 2)


        csv_path = os.path.splitext(path)[0]+'_loss_'+name+'.csv'
        df = pd.DataFrame(total)
        df.to_csv(csv_path)

def show_features(recording_path):


    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for n in dirs:
                subdir = os.path.join(root, n) 
                video_loc = os.path.join(subdir, n+'.mp4')
                
                if os.path.exists(video_loc):
                    videos.append(video_loc)
    
    feature_paths = [os.path.splitext(vid)[0]+'_outputs.h5' for vid in videos]

    for path in feature_paths:
        with h5py.File(path, 'r') as f:
            features = np.array(f['resnet50']['spatial_features'][:])

        a = 0
        b = len(features)-1
        jump = 10
        diffs = [(i,i+jump) for i in range(a,b-jump,jump)]

    
        diffs = [np.sum(np.absolute(features[b]-features[a])) for a,b in diffs]

        print(path)

        plt.plot(diffs)
        plt.show()




def infer_loss_specialists(config_path, recording_path, behaviors):

    model_paths = [config_path]
    for b in behaviors:
        model_paths.append(os.path.join(os.path.split(model_paths[0])[0], 'specialists\\'+os.path.splitext(os.path.split(model_paths[0])[1])[0]+'_'+b+'.yaml'))

    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for n in dirs:
                subdir = os.path.join(root, n) 
                video_loc = os.path.join(subdir, n+'.mp4')
                
                if os.path.exists(video_loc):
                    videos.append(video_loc)
    
    encoded_feature_paths = [os.path.splitext(vid)[0]+'_outputs_encoded.h5' for vid in videos]

    feats = {vid:None for vid in encoded_feature_paths}



    for path in encoded_feature_paths:
        complete = []

        for model_path in model_paths:

            print(model_path)

        

            with open(model_path, 'r') as file:
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

            if feats[path] is None:
                with h5py.File(path, 'r') as f:
                    features = np.array(f['features'])

                # make the indices
                indices = [0,4]
                
                reverse = [i*-1 for i in indices[::-1] if i != 0]
                reverse.extend(indices)


                data = features
                inputs = []
                targets = []
                for i in range(reverse[-1], len(data)-reverse[-1]-1):
                    X = [data[i+j,:] for j in reverse]
                    X = np.array(X)

                    Y = [data[i+j,:] for j in range(reverse[0],reverse[-1]+1)]
                    Y = np.array(Y)

                    X_prime = np.zeros_like(Y)

                    for r in range(len(X_prime[0,:])):
                        row = np.interp([-4,-3,-2,-1,0,1,2,3,4], [-4,0,4], X[:,r])

                        X_prime[:,r] = row
                    
                    # X_prime = torch.from_numpy(X_prime)
                    # Y = torch.from_numpy(Y)

                    inputs.append(X_prime)
                    targets.append(Y)
                feats[path] = (inputs, targets)

            inputs, targets = feats[path]

            loss = []

            for i in range(0, len(inputs), 2000):
                start = i 
                end = i+2000

                if end>len(inputs):
                    end = len(inputs)

                X1 = torch.from_numpy(np.array(inputs[start:end]))

                Y1 = np.array(targets[start:end])


                # Move X to the same device as the model
                X1 = X1.to(device)

                with torch.no_grad():
                    # Assuming 'input_data' is the data you want to predict on
                    # Make sure it's processed as your model expects
                    # For example, you might need to add a batch dimension using input_data.unsqueeze(0)

                    predictions = autoencoder(X1)

                    predictions = predictions.cpu()

                    predictions = predictions.numpy()

                for p in range(0,len(predictions)):
                    mse = (np.square(Y1[p] - predictions[p])).mean()
                    loss.append(mse)

            total = []


            for i in range(0, indices[-1]):
                total.append(loss[0])
            for i in range(indices[-1], len(data)-indices[-1]-1):
                total.append(loss[i-indices[-1]])
            for i in range(len(data)-indices[-1]-1, len(data)):
                total.append(loss[-1])

            if len(data)!=len(total):
                raise Exception('Lengths do not match!')
            

            total = np.array(total)

            complete.append(total)

            #total = smooth(total, 2)

        complete = np.array(complete)
        complete = np.transpose(complete)

        csv_path = os.path.splitext(path)[0]+'_loss.csv'
        df = pd.DataFrame(complete)
        df.to_csv(csv_path)
        print(f'finished with {path}')




def infer(config_path, recording_path):
    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for n in dirs:
                subdir = os.path.join(root, n) 
                video_loc = os.path.join(subdir, n+'.mp4')
                
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
        indices = [0,4]
        
        reverse = [i*-1 for i in indices[::-1] if i != 0]
        reverse.extend(indices)


        data = features
        inputs = []
        targets = []
        for i in range(reverse[-1], len(data)-reverse[-1]-1):
            X = [data[i+j,:] for j in reverse]
            X = np.array(X)

            Y = [data[i+j,:] for j in range(reverse[0],reverse[-1]+1)]
            Y = np.array(Y)

            X_prime = np.zeros_like(Y)

            for r in range(len(X_prime[0,:])):
                row = np.interp([-4,-3,-2,-1,0,1,2,3,4], [-4,0,4], X[:,r])

                X_prime[:,r] = row
            
            # X_prime = torch.from_numpy(X_prime)
            # Y = torch.from_numpy(Y)

            inputs.append(X_prime)
            targets.append(Y)


        loss = []

        for i in range(0, len(inputs), 1000):
            start = i 
            end = i+1000

            if end>len(inputs):
                end = len(inputs)

            X1 = torch.from_numpy(np.array(inputs[start:end]))
            print(X1.shape)

            Y1 = np.array(targets[start:end])


            # Move X to the same device as the model
            X1 = X1.to(device)

            with torch.no_grad():
                # Assuming 'input_data' is the data you want to predict on
                # Make sure it's processed as your model expects
                # For example, you might need to add a batch dimension using input_data.unsqueeze(0)

                predictions = autoencoder(X1)

                predictions = predictions.cpu()

                predictions = predictions.numpy()

            for p in range(0,len(predictions)):
                mse = (np.square(Y1[p] - predictions[p])).mean()
                loss.append(mse)

        total = []


        for i in range(0, indices[-1]):
            total.append(loss[0])
        for i in range(indices[-1], len(data)-indices[-1]-1):
            total.append(loss[i-indices[-1]])
        for i in range(len(data)-indices[-1]-1, len(data)):
            total.append(loss[-1])

        if len(data)!=len(total):
            raise Exception('Lengths do not match!')
        
        print(f'finished with {path}')

        total = np.array(total)

        #total = smooth(total, 2)


        csv_path = os.path.splitext(path)[0]+'_loss'+'.csv'
        df = pd.DataFrame(total)
        df.to_csv(csv_path)


            