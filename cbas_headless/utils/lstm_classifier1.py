import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import h5py
import random
import yaml
from sklearn.metrics import classification_report

class S_Features(Dataset):
    def __init__(self, instance_paths, seq_len, behaviors_ordered=None, test_set=.2):
        self.behaviors = behaviors_ordered or []
        instances = []

        for instance_path in instance_paths:
            with open(instance_path) as file:
                training_set = yaml.safe_load(file)
                behaviors = training_set['behaviors']
                self.behaviors.extend(b for b in behaviors if b not in self.behaviors)

                for b in behaviors:
                    for inst in training_set['instances'][b]:
                        coin_flip = random.random()
                        if coin_flip > test_set:
                            instances.append(inst)

        self.instances = instances
        self.seq_len = seq_len
        self.data = self.load_data()
        self.split_data(test_set)

    def load_data(self):
        data = []
        for inst in self.instances:
            label = inst['label']
            video = inst['video']
            start = inst['start']
            end = inst['end']

            file_path = os.path.splitext(video)[0] + "_outputs_encoded_features.h5"

            with h5py.File(file_path, 'r') as f:
                length = len(np.array(f['features']))
                abs_start = 0
                abs_end = length

                random_ind = random.randint(start, end)

                seq_start = random_ind - self.seq_len
                seq_end = random_ind + self.seq_len + 1
                seq_start = max(abs_start, seq_start)
                seq_end = min(abs_end, seq_end)

                sequence = np.array(f['features'][seq_start:seq_end])

                if len(sequence) < 2 * self.seq_len + 1:
                    sequence_prime = [np.zeros_like(sequence[0])] * (2 * self.seq_len + 1 - len(sequence))
                    sequence_prime.extend(sequence)
                    sequence = np.array(sequence_prime)

                # Some random padding of the sequence
                coin_flip = random.random()
                if coin_flip < 0.05:
                    amount = random.randint(1, self.seq_len)
                    sequence[:amount] = np.zeros_like(sequence[0])
                elif coin_flip > 0.95:
                    amount = random.randint(1, self.seq_len)
                    sequence[-amount:] = np.zeros_like(sequence[0])

                data.append((sequence, label))

        return data

    def split_data(self, test_set):
        split_point = int(len(self.data) * (1 - test_set))
        self.train_data, self.test_data = self.data[:split_point], self.data[split_point:]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sequence, label = self.train_data[idx]
        onehot = np.zeros(len(self.behaviors), dtype=np.float32)
        onehot[self.behaviors.index(label)] = 1.0
        return sequence, onehot

    def get_test_set(self):
        test_sequences, test_labels = zip(*self.test_data)
        test_onehots = np.zeros((len(test_sequences), len(self.behaviors)), dtype=np.float32)
        for i, label in enumerate(test_labels):
            test_onehots[i, self.behaviors.index(label)] = 1.0
        return test_sequences, test_onehots

class LSTM_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, d_model, num_classes, dropout=0.1):
        super(LSTM_classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.l1 = nn.Linear(hidden_size, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.l1(output[:, -1, :])
        output = self.dropout1(F.relu(output))
        output = self.l2(output)
        output = self.dropout2(F.relu(output))
        output = self.out(output)

        output = F.softmax(output, dim=1)
        return output

def train(train_set, test_set, seq_len, classes):
    input_dim = 128
    hidden_size = 256
    d_model = 512
    lr = 0.01
    num_epochs = 1000

    m_path = 'lstm_classifier.pth'
    config_path = os.path.splitext(m_path)[0] + '.yaml'

    model_config = {
        'seq_length': seq_len,
        'input_dim': input_dim,
        'hidden_size': hidden_size,
        'd_model': d_model,
        'learning_rate': lr,
        'model_path': m_path
    }

    with open(config_path, 'w') as file:
        yaml.dump(model_config, file)

    lstm = LSTM_classifier(input_dim, hidden_size=hidden_size, d_model=d_model, num_classes=len(classes))
    optimizer_total = torch.optim.Adam(lstm.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lstm.to(device)

    for epoch in range(num_epochs):
        for seq, target in train_set:
            seq, target = seq.to(device), target.to(device)
            optimizer_total.zero_grad()
            output = lstm(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer_total.step()

        if epoch % 10 == 0:
            for seq, target in test_set:
                seq, target = seq.to(device), target.to(device)
                output = lstm(seq)
                preds = []
                trues = []
                pred_indices = output.argmax(dim=1).detach().cpu().numpy()
                true_indices = target.argmax(dim=1).detach().cpu().numpy()
                preds.extend(pred_indices)
                trues.extend(true_indices)
                report = classification_report(trues, preds)
                print(report)

        if (epoch + 1) % 50 == 0:
            torch.save(lstm, m_path)

def load_data(instance_paths, seq_len, behaviors_ordered=None, test=0.2):
    s_dataset = S_Features(instance_paths=instance_paths, seq_len=seq_len, behaviors_ordered=behaviors_ordered)
    train_len = int(len(s_dataset) * (1 - test))
    train_set, test_set = random_split(s_dataset, [train_len, len(s_dataset) - train_len])
    training_set = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=8)
    testing_set = DataLoader(test_set, batch_size=1024, shuffle=True, num_workers=8)
    train(training_set, testing_set, seq_len=seq_len, classes=s_dataset.behaviors)
