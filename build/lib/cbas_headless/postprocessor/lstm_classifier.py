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
from torch.utils.data import random_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from . import frame_encoder

import pickle

import pandas as pd

class S_Features(Dataset):
    def __init__(self, instance_paths, seq_len, behaviors_ordered=None, test_set=.2, encoder=None):

        self.encoder = encoder 
        
        self.behaviors = behaviors_ordered
        if self.behaviors is None:
            self.behaviors = []
        instances = []
        test_instances = []

        for recording, insts in instance_paths.items():
            
            with open(insts) as file:
                training_set = yaml.safe_load(file)

                behaviors = training_set['behaviors']
                for b in behaviors:
                    if b not in self.behaviors:
                        if behaviors_ordered is None:
                            self.behaviors.append(b)
                        else:
                            continue
                    for inst in training_set['instances'][b]:
                        inst['deg_features'] = os.path.join(recording, os.path.splitext(os.path.split(inst['video'])[1])[0], os.path.splitext(os.path.split(inst['video'])[1])[0]+"_outputs.h5")
                        
                        
                        if not os.path.exists(inst['deg_features']):
                            print(f'Could not find {inst["deg_features"]}, skipping')
                            continue

                        coin_flip = random.random()
                        if coin_flip>test_set:
                            instances.append(inst)
                        else:
                            test_instances.append(inst)


        self.instances = instances
        self.test_instances = test_instances
        self.seq_len = seq_len


        encodings = {inst['deg_features']:None for inst in self.instances}
        if encoder is not None:
            encodings = frame_encoder.encode(encoder, encodings)
        self.encodings = encodings


        self.seq_labels = []
            
        self.generate()

        random.shuffle(self.seq_labels)
        random.shuffle(self.seq_labels)
        random.shuffle(self.seq_labels)


        print(len(self.seq_labels))

    def generate(self):
        print('generating')
        for i in range(len(self.instances)):
            inst = self.instances[i]


            label = inst['label']
            video = inst['video']
            start = inst['start']
            end = inst['end']

            for t in range(int(start), int(end)):


                file_path = os.path.splitext(inst['deg_features'])[0]+"_encoded_features.h5"
                outputs_path = inst['deg_features']
                data = None

                if not os.path.exists(file_path) and self.encoder is None:
                    print(f'Could not find {file_path}, skipping')
                    print(f'Either set encoding to true or infer with the encoder separately before training.')
                    continue
                elif not os.path.exists(file_path) and self.encoder is not None:
                    data = self.encodings[outputs_path]

                if data is None:
                    with h5py.File(file_path, 'r') as f:
                        length = len(np.array(f['features']))

                        before_buffer = True
                        after_buffer = True

                        abs_start = 0 
                        abs_end = length
                        

                        seq_start = t - self.seq_len
                        seq_end = t + self.seq_len+1
                        if seq_start<abs_start:
                            before_buffer = False
                            seq_start = abs_start
                        if seq_end>abs_end:
                            after_buffer = False
                            seq_end = abs_end

                        sequence = np.array(f['features'][seq_start:seq_end])
                        with h5py.File(outputs_path, 'r') as f:
                            logits = np.array(f['resnet50']['logits'][seq_start:seq_end])
                            sequence = np.concatenate((sequence, logits), axis=1)

                        if len(sequence) < 2*self.seq_len+1:
                            sequence_prime = []
                            if not before_buffer:
                                diff = 2*self.seq_len+1 - len(sequence)
                                for j in range(diff):
                                    sequence_prime.append(np.zeros_like(sequence[0]))
                                for s in sequence:
                                    sequence_prime.append(s)
                            elif not after_buffer:
                                diff = 2*self.seq_len+1 - len(sequence)
                                for s in sequence:
                                    sequence_prime.append(s)
                                for j in range(diff):
                                    sequence_prime.append(np.zeros_like(sequence[0]))
                            
                            sequence = np.array(sequence_prime)
                        
                        # final check to make sure it's the right length
                        if len(sequence) != 2*self.seq_len+1:
                            print('found the incorrect sequence length, trying the next one')
                            continue

                        # some random padding of the sequence to make it so that the dataset is bit more balanced
                        coin_flip = random.random()
                        if coin_flip<.01:
                            # ablate some of the sequence to mimic starting at the beginning of a video
                            amount = random.randint(1,self.seq_len)

                            for j in range(amount):
                                sequence[j] = np.zeros_like(sequence[0])

                        elif coin_flip>.99:
                            # ablate some of the sequence to mimic starting at the end of a video
                            amount = random.randint(1,self.seq_len)

                            for j in range(amount):
                                sequence[len(sequence)-j-1] = np.zeros_like(sequence[0])
                    
                    # almost done, now just generate a one-hot encoding of the label
                    onehot = []
                    for b in self.behaviors:
                        if b!=label:
                            onehot.append(0.0)
                        else:
                            onehot.append(1.0)
                    
                    onehot = np.array(onehot, dtype="float32")

                    # return sequence and label
                    self.seq_labels.append((sequence, onehot)) 
                else:
                    length = len(np.array(data))

                    before_buffer = True
                    after_buffer = True

                    abs_start = 0 
                    abs_end = length
                    

                    seq_start = t - self.seq_len
                    seq_end = t + self.seq_len+1
                    if seq_start<abs_start:
                        before_buffer = False
                        seq_start = abs_start
                    if seq_end>abs_end:
                        after_buffer = False
                        seq_end = abs_end

                    sequence = np.array(data[seq_start:seq_end])
                    with h5py.File(outputs_path, 'r') as f:
                        logits = np.array(f['resnet50']['logits'][seq_start:seq_end])
                        sequence = np.concatenate((sequence, logits), axis=1)

                    if len(sequence) < 2*self.seq_len+1:
                        sequence_prime = []
                        if not before_buffer:
                            diff = 2*self.seq_len+1 - len(sequence)
                            for j in range(diff):
                                sequence_prime.append(np.zeros_like(sequence[0]))
                            for s in sequence:
                                sequence_prime.append(s)
                        elif not after_buffer:
                            diff = 2*self.seq_len+1 - len(sequence)
                            for s in sequence:
                                sequence_prime.append(s)
                            for j in range(diff):
                                sequence_prime.append(np.zeros_like(sequence[0]))
                        
                        sequence = np.array(sequence_prime)
                    
                    # final check to make sure it's the right length
                    if len(sequence) != 2*self.seq_len+1:
                        print('found the incorrect sequence length, trying the next one')
                        continue

                    # some random padding of the sequence to make it so that the dataset is bit more balanced
                    coin_flip = random.random()
                    if coin_flip<.01:
                        # ablate some of the sequence to mimic starting at the beginning of a video
                        amount = random.randint(1,self.seq_len)

                        for j in range(amount):
                            sequence[j] = np.zeros_like(sequence[0])

                    elif coin_flip>.99:
                        # ablate some of the sequence to mimic starting at the end of a video
                        amount = random.randint(1,self.seq_len)

                        for j in range(amount):
                            sequence[len(sequence)-j-1] = np.zeros_like(sequence[0])
                
                # almost done, now just generate a one-hot encoding of the label
                onehot = []
                for b in self.behaviors:
                    if b!=label:
                        onehot.append(0.0)
                    else:
                        onehot.append(1.0)
                
                onehot = np.array(onehot, dtype="float32")

                # return sequence and label
                self.seq_labels.append((sequence, onehot)) 


    def __len__(self):
        return len(self.seq_labels)

    def __getitem__(self, idx):
        return self.seq_labels[idx]


class S_Features_Test(Dataset):
    def __init__(self, dataset):
        
        self.behaviors = dataset.behaviors

        self.instances = dataset.test_instances
        self.seq_len = dataset.seq_len

        self.encoder = dataset.encoder

        encodings = {inst['deg_features']:None for inst in self.instances}
        if self.encoder is not None:
            encodings = frame_encoder.encode(self.encoder, encodings)
        self.encodings = encodings

        self.seq_labels = []
        self.generate()

    def generate(self):
        print('generating')
        for i in range(10000):
            iidx = i%len(self.instances)
            inst = self.instances[iidx]


            label = inst['label']
            video = inst['video']
            start = inst['start']
            end = inst['end']



            file_path = os.path.splitext(inst['deg_features'])[0]+"_encoded_features.h5"
            outputs_path = inst['deg_features']
            data = None

            if not os.path.exists(file_path) and self.encoder is None:
                print(f'Could not find {file_path}, skipping')
                print(f'Either set encoding to true or infer with the encoder separately before training.')
                continue
            elif not os.path.exists(file_path) and self.encoder is not None:
                data = self.encodings[outputs_path]

            if data is None:

                with h5py.File(file_path, 'r') as f:
                    length = len(np.array(f['features']))

                    before_buffer = True
                    after_buffer = True

                    abs_start = 0 
                    abs_end = length
                    
                    random_ind = random.randint(start, end)

                    seq_start = random_ind - self.seq_len
                    seq_end = random_ind + self.seq_len+1
                    if seq_start<abs_start:
                        before_buffer = False
                        seq_start = abs_start
                    if seq_end>abs_end:
                        after_buffer = False
                        seq_end = abs_end

                    sequence = np.array(f['features'][seq_start:seq_end])
                    with h5py.File(outputs_path, 'r') as f:
                        logits = np.array(f['resnet50']['logits'][seq_start:seq_end])
                        sequence = np.concatenate((sequence, logits), axis=1)
                        #sequence = logits

                    if len(sequence) < 2*self.seq_len+1:
                        sequence_prime = []
                        if not before_buffer:
                            diff = 2*self.seq_len+1 - len(sequence)
                            for i in range(diff):
                                sequence_prime.append(np.zeros_like(sequence[0]))
                            for s in sequence:
                                sequence_prime.append(s)
                        elif not after_buffer:
                            diff = 2*self.seq_len+1 - len(sequence)
                            for s in sequence:
                                sequence_prime.append(s)
                            for i in range(diff):
                                sequence_prime.append(np.zeros_like(sequence[0]))
                        
                        sequence = np.array(sequence_prime)
                    
                    # final check to make sure it's the right length
                    if len(sequence) != 2*self.seq_len+1:
                        print('found the incorrect sequence length, trying the next one')
                        continue

                    # some random padding of the sequence to make it so that the dataset is bit more balanced
                    coin_flip = random.random()
                    if coin_flip<.05:
                        # ablate some of the sequence to mimic starting at the beginning of a video
                        amount = random.randint(1,self.seq_len)

                        for i in range(amount):
                            sequence[i] = np.zeros_like(sequence[0])

                    elif coin_flip>.95:
                        # ablate some of the sequence to mimic starting at the end of a video
                        amount = random.randint(1,self.seq_len)

                        for i in range(amount):
                            sequence[len(sequence)-i-1] = np.zeros_like(sequence[0])
                
                # almost done, now just generate a one-hot encoding of the label
                onehot = []
                for b in self.behaviors:
                    if b!=label:
                        onehot.append(0.0)
                    else:
                        onehot.append(1.0)
                
                onehot = np.array(onehot, dtype="float32")

                # return sequence and label
                self.seq_labels.append((sequence, onehot)) 
            else:
                length = len(np.array(data))

                before_buffer = True
                after_buffer = True

                abs_start = 0 
                abs_end = length
                
                random_ind = random.randint(start, end)

                seq_start = random_ind - self.seq_len
                seq_end = random_ind + self.seq_len+1
                if seq_start<abs_start:
                    before_buffer = False
                    seq_start = abs_start
                if seq_end>abs_end:
                    after_buffer = False
                    seq_end = abs_end

                sequence = np.array(data[seq_start:seq_end])
                with h5py.File(outputs_path, 'r') as f:
                    logits = np.array(f['resnet50']['logits'][seq_start:seq_end])
                    sequence = np.concatenate((sequence, logits), axis=1)
                    #sequence = logits

                if len(sequence) < 2*self.seq_len+1:
                    sequence_prime = []
                    if not before_buffer:
                        diff = 2*self.seq_len+1 - len(sequence)
                        for i in range(diff):
                            sequence_prime.append(np.zeros_like(sequence[0]))
                        for s in sequence:
                            sequence_prime.append(s)
                    elif not after_buffer:
                        diff = 2*self.seq_len+1 - len(sequence)
                        for s in sequence:
                            sequence_prime.append(s)
                        for i in range(diff):
                            sequence_prime.append(np.zeros_like(sequence[0]))
                    
                    sequence = np.array(sequence_prime)
                
                # final check to make sure it's the right length
                if len(sequence) != 2*self.seq_len+1:
                    print('found the incorrect sequence length, trying the next one')
                    continue

                # some random padding of the sequence to make it so that the dataset is bit more balanced
                coin_flip = random.random()
                if coin_flip<.05:
                    # ablate some of the sequence to mimic starting at the beginning of a video
                    amount = random.randint(1,self.seq_len)

                    for i in range(amount):
                        sequence[i] = np.zeros_like(sequence[0])

                elif coin_flip>.95:
                    # ablate some of the sequence to mimic starting at the end of a video
                    amount = random.randint(1,self.seq_len)

                    for i in range(amount):
                        sequence[len(sequence)-i-1] = np.zeros_like(sequence[0])
            
            # almost done, now just generate a one-hot encoding of the label
            onehot = []
            for b in self.behaviors:
                if b!=label:
                    onehot.append(0.0)
                else:
                    onehot.append(1.0)
            
            onehot = np.array(onehot, dtype="float32")

            # return sequence and label
            self.seq_labels.append((sequence, onehot)) 


    def __len__(self):
        return len(self.seq_labels)

    def __getitem__(self, idx):
        return self.seq_labels[idx]
        
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x):
        return x + self.positional_encodings


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3):
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
        src2 = F.relu(self.linear1(src2))
        src2 = self.dropout2(src2)
        src2 = self.linear2(src2)

        src = src + self.dropout(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3):
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
        
        input2 = F.relu(self.linear1(input2))
        input2 = self.dropout3(input2)
        input2 = self.linear2(input2)
        
        output = input + self.dropout2(input2)
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model, nhead, num_encoder_layers, seq_len):
        super(Encoder, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_embedding = nn.Linear(input_dim, d_model)

        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)])

        
        self.dropout = nn.Dropout(0.2)

        self.encoder_to_latent = nn.Linear(d_model, latent_dim)   
        

    def forward(self, src):
        src = self.encoder_embedding(src)
        src = self.pos_encoder(src)  

        for layer in self.encoder_layers:
            src = layer(src)

        src = self.encoder_to_latent(src)

        return src
    
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, d_model, nhead, num_decoder_layers, seq_len, dropout=0.2):
        super(Decoder, self).__init__()

        self.latent_to_decoder = nn.Linear(latent_dim, d_model)  
        self.input_embedding = nn.Linear(input_dim, d_model)  

        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)

        
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


        self.lstm.flatten_parameters()  
        input, _ = self.lstm(input)
        input = input[:, -1, :]

        output = self.output_layer(input)

        return output
    
 
class Classifier(nn.Module):
    def __init__(self, encoder, decoder):
        super(Classifier, self).__init__()

        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, src):

        latent = self.encoder(src)
        output = self.decoder(src, latent)

        return output
    
# class LSTM_classifier(nn.Module):
#     def __init__(self, input_size, hidden_size, d_model, num_classes, dropout=0.1):
#         super(LSTM_classifier, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
#         self.l1 = nn.Linear(hidden_size, d_model)
#         self.dropout1 = nn.Dropout(p=dropout)
#         self.l2 = nn.Linear(d_model, d_model)
#         self.dropout2 = nn.Dropout(p=dropout)
#         self.out = nn.Linear(d_model, num_classes)

#     def forward(self, x):
#         output, _ = self.lstm(x)
#         output = self.l1(output[:, -1, :])
#         output = self.dropout1(F.relu(output))
#         output = self.l2(output)
#         output = self.dropout2(F.relu(output))
#         output = self.out(output)

#         return output



def train(m_path, train_set, test_set, seq_len, classes):
    
    input_dim = 137
    hidden_size = 128
    latent_dim = 64
    d_model = 64
    lr = .00001
    nheads = 16
    num_encoder_layers = 5
    num_decoder_layers = 5
    num_epochs = 50

    config_path = os.path.splitext(m_path)[0]+'.yaml'

    model_config = {
        'seq_length':seq_len,
        'input_dim':input_dim,
        'hidden_size': hidden_size,
        'd_model': d_model,
        'learning_rate':lr,
        'model_path':m_path
    }

    with open(config_path, 'w') as file:
        yaml.dump(model_config, file)


    encoder = Encoder(input_dim, latent_dim, d_model, nheads, num_encoder_layers, seq_len*2+1)
    decoder = Decoder(input_dim, latent_dim, len(classes), d_model, nheads, num_decoder_layers, seq_len*2+1)
    classifier = Classifier(encoder, decoder)
    optimizer_total = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    classifier.to(device)

    # Training loop
    for epoch in range(num_epochs):

        for seq, target in train_set:

            print(f'Epoch {epoch}:')

            # Move data to the device
            target = target.long()
            target = torch.argmax(target, dim=1)
            seq, target = seq.to(device), target.to(device)

            # Forward pass, backward pass, and optimize
            optimizer_total.zero_grad()
            output = classifier(seq)
            loss = criterion(output, target)

            loss.backward()
            optimizer_total.step()

            print(f'Training Loss: {loss.item()}')

        if (epoch+1)%3==0:
        
            # Initialize empty lists to store predicted and true labels
            all_preds = []
            all_trues = []

            for seq, target in test_set:

                # Move data to the device
                target = target.long()
                target = torch.argmax(target, dim=1)
                seq, target = seq.to(device), target.to(device)
                output = classifier(seq)

                # Convert predicted and target values to integers
                pred_indices = output.argmax(dim=1).detach().cpu().numpy()
                true_indices = target.detach().cpu().numpy()

                # Append indices to the lists
                all_preds.extend(pred_indices)
                all_trues.extend(true_indices)


            report = classification_report(all_trues, all_preds)
            print(report)



        

        if (epoch+1)%3==0:
            torch.save(classifier, m_path)



def load_data(instance_paths, cache_path, seq_len, behaviors_ordered=None, test=.2, cache=True, encoder=None):

    testing = os.path.join(cache_path, 'testing.pkl')
    training = os.path.join(cache_path, 'training.pkl')

    if not cache:
        # If not caching, create and save the dataset objects
        s_dataset = S_Features(instance_paths=instance_paths, seq_len=seq_len, behaviors_ordered=behaviors_ordered, encoder=encoder, test_set=test)
        s_dataset_test = S_Features_Test(s_dataset)

        with open(training, 'wb') as file:
            pickle.dump(s_dataset, file)
        with open(testing, 'wb') as file:
            pickle.dump(s_dataset_test, file)
    else:
        # If caching, load the dataset objects from the pickle files
        with open(training, 'rb') as file:
            s_dataset = pickle.load(file)
        with open(testing, 'rb') as file:
            s_dataset_test = pickle.load(file)

    # Create a DataLoader
    training_set = DataLoader(s_dataset, batch_size=128, shuffle=True, num_workers=8)
    testing_set = DataLoader(s_dataset_test, batch_size=128, shuffle=True, num_workers=8)

    return training_set, testing_set, training, testing, s_dataset.behaviors


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
        pppath = pconfig['postprocessors']
        ppconfig = pconfig['postprocessors_config']
        # extract the mdoels config file
        try:
            with open(ppconfig, 'r') as file:
                ppconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return (pppath, ppconfig)
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
        pppath = pconfig['postprocessors']
        ppconfig = pconfig['postprocessors_config']
        # extract the mdoels config file
        try:
            with open(ppconfig, 'r') as file:
                ppconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return (pppath, ppconfig)
    
def load_recording_paths(project_config='undefined'):
    # open the project config and get the test_set yaml path
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

        # grabbing the locations of the recordings
        recordings_path = pconfig['recordings_path']

    
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

        # grabbing the locations of the recordings
        recordings_path = pconfig['recordings_path']
    
    return recordings_path

def load_set_paths(project_config='undefined'):
    # open the project config and get the test_set yaml path
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

        # grabbing the locations of the recordings
        ts_path = pconfig['testsets_path']
        trs_path = pconfig['trainingsets_path']

    
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

        # grabbing the locations of the recordings
        ts_path = pconfig['testsets_path']
        trs_path = pconfig['trainingsets_path']
    
    return trs_path, ts_path




def train_lstm(recording_names, deg_name, postprocessor_name, seq_len, classes=None, encode=True, load_from_cache=False, project_config="undefined"):

    trs_path, ts_path = load_set_paths(project_config)
    recordings_path = load_recording_paths(project_config)
    pppath, ppconfig = load_postprocessors(project_config)

    pps = list(ppconfig['models'].keys())

    if postprocessor_name not in pps:
        raise Exception('Postprocessor encoder not found, train an encoder first before training this model.')
    
    mconfig = ppconfig['models'][postprocessor_name]

    encoder = os.path.splitext(mconfig['encoder'])[0]+'.yaml'

    instance_paths = {}

    for rn in recording_names:
        if not os.path.exists(os.path.join(trs_path, rn+"_training.yaml")):
            raise Exception(f'Instance path not found for recording {rn} could not be found.')
        
        recording = os.path.join(recordings_path, rn, deg_name)
        if not os.path.exists(recording):
            raise Exception(f'Recording deg outputs for {recording}, {deg_name} could not be found.')

        instance_paths[recording] = os.path.join(trs_path, rn+"_training.yaml")
        


    # Load the data
    cache_path = os.path.join(ts_path, postprocessor_name)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    trainset, testset, trsl, tsl, behaviors = load_data(instance_paths, cache_path, seq_len, classes, cache=load_from_cache, encoder=encoder)

    classifier_path = os.path.join(pppath, postprocessor_name, 'lstm_classifier.pth')

    # Train the model
    train(classifier_path, trainset, testset, seq_len, behaviors)


    
    ppconfig['models'][postprocessor_name]['training_set'] = trsl
    ppconfig['models'][postprocessor_name]['test_set'] = tsl
    ppconfig['models'][postprocessor_name]['classifier'] = classifier_path
    ppconfig['models'][postprocessor_name]['behaviors'] = behaviors


    with open(os.path.join(pppath, 'postprocessors.yaml'), 'w') as file:
        yaml.dump(ppconfig, file)




def infer(config_path, recording_path, videos):
    
    deg_feature_paths = [os.path.splitext(vid)[0]+'_outputs.h5' for vid in videos]
    feature_paths = [os.path.splitext(vid)[0]+'_outputs_encoded_features.h5' for vid in videos]

    with open(config_path, 'r') as file:
        model_config = yaml.safe_load(file)


    # model_config = {
    #     'seq_length':seq_len,
    #     'input_dim':input_dim,
    #     'model_dim':d_model,
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

    seq_len = model_config['seq_length']

    for path1, path2 in zip(feature_paths, deg_feature_paths):

        with h5py.File(path1, 'r') as f:
            features = np.array(f['features'][:])
        with h5py.File(path2, 'r') as f:
            logits = np.array(f['resnet50']['logits'][:])

            features = np.concatenate((features, logits), axis=1)



        data = features
        inputs = []
        for i in range(0, len(data)):
            start = i-seq_len
            end = i+seq_len+1

            begin_append = 0
            end_append = 0

            if start<0:
                begin_append = -start
                start = 0
            if end>len(data):
                end_append = end - len(data)
                end = len(data)

            X = data[start:end,:]
            if begin_append!=0 or end_append!=0:
                X_prime = []
                for i in range(begin_append):
                    X_prime.append(np.zeros_like(X[0]))
                X_prime.extend(X)
                for i in range(end_append):
                    X_prime.append(np.zeros_like(X[0]))
                
                X = np.array(X_prime)

            inputs.append(X)


        total = []
        for i in range(0, len(inputs), 1000):
            start = i 
            end = i+1000

            if end>len(inputs):
                end = len(inputs)

            X1 = torch.from_numpy(np.array(inputs[start:end]))


            # Move X to the same device as the model
            X1 = X1.to(device)

            with torch.no_grad():
                # Assuming 'input_data' is the data you want to predict on
                # Make sure it's processed as your model expects
                # For example, you might need to add a batch dimension using input_data.unsqueeze(0)

                predictions = autoencoder(X1)

                # Apply softmax to obtain class probabilities
                predictions_prob = torch.softmax(predictions, dim=1)

                # Convert class probabilities to one-hot encoded representations
                _, predicted_classes = predictions_prob.max(dim=1)
                one_hot_predictions = torch.zeros_like(predictions_prob)
                one_hot_predictions.scatter_(1, predicted_classes.view(-1, 1), 1)

                # # Convert one-hot predictions to NumPy array
                one_hot_predictions = one_hot_predictions.cpu().numpy()

                predictions_prob = predictions_prob.cpu().numpy()

                total.extend(one_hot_predictions)


        if len(data)!=len(total):
            total = total[:len(data)]
        
        print(f'finished with {path2}')

        total = np.array(total)


        csv_path = os.path.splitext(path2)[0]+'_inferences'+'.csv'

        df = pd.DataFrame(total)
        df.to_csv(csv_path)









            