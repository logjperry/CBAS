from sklearn.ensemble import RandomForestClassifier as RF
import yaml
import random
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import xgboost as xgb

import pywt
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import UnivariateSpline

def decompose_series(series, wavelet, levels):
    coeffs = pywt.wavedec(series, wavelet, level=levels)
    cA = coeffs[0]  # Approximation coefficients
    cD = coeffs[1:]  # Detail coefficients
    return cA, cD

def create_feature_vector(cA, cD):
    features = []
    # Add features from approximation coefficients
    features.extend([cA.mean(), cA.var()])

    # # Add features from detail coefficients
    # for coeff in cD:
    #     features.extend([coeff.mean(), coeff.var(), np.sum(coeff**2)])  # Example features

    return np.array(features)

def downsample_and_average_spline_multifeature(time_series, n_samples, n_shifts, smoothing_factor):
    """
    Downsamples and averages spline fits of a multi-feature time series.

    Parameters:
    time_series (np.array): The original time series data with shape Tx64.
    n_samples (int): Number of samples to downsample to.
    n_shifts (int): Number of shifts for downsampling.
    smoothing_factor (float): Smoothing factor for spline fitting.

    Returns:
    np.array: The averaged spline fit time series with shape Tx64.
    """
    T, num_features = time_series.shape
    averaged_spline_fit = np.zeros_like(time_series)
    time_points = np.linspace(0, T - 1, T)

    for feature in range(num_features):
        feature_series = time_series[:, feature]
        averaged_feature_fit = np.zeros(T)

        for shift in range(n_shifts):
            indices = np.linspace(shift, T - 1, n_samples, dtype=int)
            downsampled_time = time_points[indices]
            downsampled_series = feature_series[indices]

            spline = UnivariateSpline(downsampled_time, downsampled_series, s=smoothing_factor)
            averaged_feature_fit += spline(time_points)

        averaged_spline_fit[:, feature] = averaged_feature_fit / n_shifts

    return averaged_spline_fit

def generate_fibonacci_derivatives(df, indices=[1,3,5,9,15]):

    

    dfs = []
    for i in indices:

        df_temp = np.zeros_like(df)

        num_rows, num_cols = df.shape

        for t in range(num_rows):
            lower_index = t - i
            upper_index = t + i

            lower_row = df[lower_index] if 0 <= lower_index < num_rows else np.zeros(num_cols)
            upper_row = df[upper_index] if 0 <= upper_index < num_rows else np.zeros(num_cols)

            if 0 <= lower_index < num_rows and 0 <= upper_index < num_rows:
                df_temp[t] = (upper_row - lower_row)/(i*2)
            else:
                df_temp[t] = np.zeros(num_cols)
        
        dfs.append(df_temp)

    return dfs

def autocorrelation(series):
    n = len(series)
    mean = np.mean(series)
    autocorr = np.correlate(series - mean, series - mean, mode='full') / np.var(series)
    return autocorr[n-1:] / np.arange(n, 0, -1)

    


def process_features(inst, behaviors):

    X = []
    y = []

    video = inst['video']
    start = inst['start']
    end = inst['end']
    
    encoded_outputs = os.path.splitext(video)[0]+'_outputs_encoded_features.h5'
    #encoded_seq = os.path.splitext(video)[0]+'_outputs_encoded_features_encoded_seq.h5'
    outputs = os.path.splitext(video)[0]+'_outputs.h5'
    # encoded_spatial_1 = os.path.splitext(video)[0]+'_outputs_5_spatial_encoded_10fps.csv'
    # encoded_flow_1 = os.path.splitext(video)[0]+'_outputs_5_flow_encoded_10fps.csv'
    # encoded_spatial_5 = os.path.splitext(video)[0]+'_outputs_51_encoded_2fps.csv'

    
    #encoded_fibonacci = os.path.splitext(video)[0]+'_outputs_10_fibonacci_dual_encoded.csv'
    #encoded_fibonacci = os.path.splitext(video)[0]+'_outputs_contrastive_encoded.csv'
    #encoded_fibonacci = os.path.splitext(video)[0]+'_outputs_transpose_encoded.csv'
    #encoded_fibonacci = os.path.splitext(video)[0]+'_outputs_10_fibonacci_dual_16_encoded.csv'

    with h5py.File(encoded_outputs, 'r') as f:
        features = np.array(f['features'])
    #with h5py.File(encoded_seq, 'r') as f:
    #    seq = np.array(f['features'])

        
    # with h5py.File(outputs, 'r') as f:
    #     sfeatures = np.array(f['resnet50']['spatial_features'][:])
    #     ffeatures = np.array(f['resnet50']['flow_features'][:])
    #     features = np.concatenate((sfeatures, ffeatures, features), axis=1)

    # dfes1 = pd.read_csv(encoded_spatial_1)
    # dfes5 = pd.read_csv(encoded_spatial_5)

    # dfef1 = pd.read_csv(encoded_flow_1)

    #dfef = pd.read_csv(encoded_fibonacci)

    # dfes1 = dfes1.to_numpy()[1:,1:]
    # dfef1 = dfef1.to_numpy()[1:,1:]
    # dfes5 = dfes5.to_numpy()[1:,1:]


    start = int(start)-1
    end = int(end)-1
    #dfef = dfef.to_numpy()[1:,1:]

    # if inst['label']=='resting':

    #     for s in range(0, len(dfef[0,:])):
    #         plt.plot(dfef[start:end,s])

    #     plt.show()

    # adj_start = start 
    # if start-20>=0:
    #     adj_start = start-20
    # adj_end = end 
    # if adj_end+20<len(dfef):
    #     adj_end = adj_end+20
    # cwt([.1*i for i in range(0,adj_end-adj_start)], dfef[adj_start:adj_end])
    
    indices=[2,4,8]

    #dfs = generate_fibonacci_derivatives(dfef, indices=indices)

    # motion = np.zeros(len(dfef[start:end,0]))

    
    # for i in range(start, end):
    #     cumsum = 0
    #     for s in range(0, len(dfef[0,:])):

    #         st = i-30
    #         en = i+31
    #         if st<0:
    #             st = 0
    #         if en>len(dfef):
    #             en = len(dfef)

    #         mean = np.mean(dfef[st:en,s])
    #         absum = np.sum(np.absolute(dfef[st:en,s]-mean))

    #         cumsum += absum
    #     motion[i-start] = cumsum

    # if inst['label']=='resting':
    #     plt.plot(motion)
    #     plt.show()



    #dfef = dfef[start:end,:]
    features = features[start:end,:]
    #seq = seq[start:end,:]

    




    # m1 = np.abs(dfs[0][start:end,:])
    # m2 = np.abs(dfs[1][start:end,:])
    # m3 = np.abs(dfs[2][start:end,:])





    index = behaviors.index(inst['label'])

    length = end-start

    for j in range(0, length):

        # ef1 = dfef1[j,:]
        # es1 = dfes1[j,:]
        # es5 = dfes5[j,:]

        #lgs = logits[j,:]
        #ef = dfef[j,:]
        feature = features[j,:]
        #s = seq[j,:]

        # m_1 = m1[j,:]
        # m_2 = m2[j,:]
        # m_3 = m3[j,:]

        # count = 1

        # if j>0:
        #     lgs+=logits[j-1,:]
        #     count+=1
        # if j<length-1:
        #     lgs+=logits[j+1,:]
        #     count+=1

        # lgs = lgs/count




        #feats = np.concatenate((feature, s), axis=0)
        feats = feature
        #features = lgs

        #features = ef

        X.append(feats)
        y.append(index)

    return (X, y)


def train(instance_paths):
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

    train_amount = .7


    training_instances = {b:[] for b in behaviors} 
    testing_instances = {b:[] for b in behaviors} 

    for b in behaviors:
        b_insts = instances[b]
        num = len(b_insts)

        num_training = int(num*train_amount)

        # shuffle the instances
        random.shuffle(b_insts)

        for i in range(0, len(b_insts)):
            if i<num_training:
                training_instances[b].append(b_insts[i])
            else:
                testing_instances[b].append(b_insts[i])

    
    # load the testing set first
    X_test = []
    y_test = []
    for b in behaviors:

        b_insts = testing_instances[b]

        len_insts = len(b_insts)
        indices = [i for i in range(0,len_insts)]
        
        for i in indices:
            inst = b_insts[i]

            X, y = process_features(inst, behaviors)

            X_test.extend(X)
            y_test.extend(y)


            
        

    trees = 1

    rf = RF(n_estimators=500, verbose=2)
    

    # Do this for the number of trees needed
    for t in range(trees):

        X = [] 
        y = []

        # load the training set
        for b in behaviors:

            b_insts = training_instances[b]

            len_insts = len(b_insts)
            indices = [i for i in range(0,len_insts)]
            
            for i in indices:
                inst = b_insts[i]
                X_train, y_train = process_features(inst, behaviors)
                            
                X.extend(X_train)
                y.extend(y_train)



        
        # Training
        rf.fit(X, y)

        predictions = rf.predict(X_test)

        report = classification_report(y_test, predictions)
        print(report)





