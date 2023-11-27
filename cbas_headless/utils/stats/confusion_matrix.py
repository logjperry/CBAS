import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score 

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd

import numpy 

import matplotlib.pyplot as plt

import matplotlib

import sys
import os

import time

import json

import yaml
from tabulate import tabulate

def precision(mat):
    if mat[1][1]+mat[0][1] == 0:
        return 1
    return mat[1][1]/(mat[1][1]+mat[0][1])

def recall(mat):
    if mat[1][1]+mat[1][0] == 0:
        return 1
    return mat[1][1]/(mat[1][1]+mat[1][0])

def specificty(mat):
    if mat[0][0]+mat[0][1] == 0:
        return 1
    return mat[0][0]/(mat[0][0]+mat[0][1])

def windowize(array, win, slide=600):
    result = np.array([0])
    c_i = 0
    for j in range(0, int(np.ceil(win/slide))):
        for i in range(j*slide, len(array)):
            if (i+1)%win == 0:
                result = np.append(result, [0])
                c_i += 1
            if array[i] == 1:
                result[c_i]=1
    return result

def f1(dirname, behaviors, val_only=True, window=1):

    

    avgScores = [0,0,0,0,0,0,0,0,0]
    dircount = 0

    conf_mat = [[],[],[],[],[],[],[],[],[]]

    multi_class = np.zeros((9,9))
    
    val_set = None
    
    with open(os.path.join(dirname,"split.yaml"), 'r') as stream:
        val_set = yaml.safe_load(stream)['val']
    print(val_set)
    if val_set==None:
        print("Validation set is empty for the specified DATA folder... Please update the split.yaml file.")
        return

    fileboutlens = []
    single = False
        
    for dir in os.listdir(dirname):

        lbname = ""
        pdname = ""

        if not os.path.isdir(dirname+"\\"+dir):
            continue
            
        #if dir not in val_set and val_only:
        #    continue
        
        

        for filename in os.listdir(dirname+"\\"+dir):
            root, ext = os.path.splitext(filename)
            if root.endswith('labels') and ext == '.csv':
                lbname = filename
            if root.endswith('predictions') and ext == '.csv':
                pdname = filename

        if lbname=="" or pdname=="":
            continue
        
        dircount+=1

        labels = pd.read_csv(dirname+"\\"+dir+"\\"+lbname)
        predictions = pd.read_csv(dirname+"\\"+dir+"\\"+pdname)

        

        t = 0
        boutlens = [[],[],[],[],[],[],[],[],[]]
        bouts = [0,0,0,0,0,0,0,0,0]
        for s in behaviors:
            actual = labels[s].to_numpy()
            count = 0
            for ind in range(0, len(actual)):
                if actual[ind] == 1:
                    count += 1
                elif count!=0:
                    boutlens[t].append(count) 
                    bouts[t] += 1 
                    count = 0
                if count>1200:
                    boutlens[t].append(count) 
                    bouts[t] += 1 
                    count = 0
            t+=1
        t = 0

        fileboutlens.append(boutlens)

        if single:
            df_avgbouts = pd.DataFrame(columns=behaviors)
            means = [0,0,0,0,0,0,0,0,0]
            for s in behaviors:
                if bouts[t]!=0:
                    counts, bins = np.histogram(np.array(boutlens[t]), bins=np.arange(0,1200,10))
                    counts = np.multiply(np.array(counts),np.array(bins[:-1]))
                    plt.clf()

                    mean = np.sum(np.multiply(np.array(counts), np.array(bins[:-1])))/np.sum(np.array(counts))
                    #counts = counts/np.sum(counts)
                    plt.stairs(counts, bins,fill=True)
                    plt.axvline(mean, color='red')

                    plt.ylabel('total frame counts')
                    plt.xlabel('bout length') 

                    plt.title('most probable bout length of a given '+s+' frame')
                    plt.savefig(os.path.join(os.getcwd(), 'bouts/'+s+".jpg"))

                    print(s+" - "+str(mean))
                    means[t] = mean

                t+=1
            df_avgbouts = pd.DataFrame(np.reshape(np.array(means),(1,9)), columns=behaviors)
            
            df_avgbouts.to_csv(os.path.join(os.getcwd(), 'bouts/avgbouts.csv'), index=False)
        

        #multi-class 
        t = 0
        for s in behaviors:
            actual = labels[s].to_numpy()

            #len(actual)
            for ind in range(0, 10000):
                if actual[ind] == 1:
                    c = 0
                    for s in behaviors:
                        pred = predictions[s].to_numpy()
                        if pred[ind] == 1:
                            multi_class[c,t] += 1
                        c+=1
            t+=1
    

        no_bg = False
        i = 0
        for s in behaviors:


            #define array of actual classes
            actual = labels[s].to_numpy()

            #define array of predicted classes
            pred = predictions[s].to_numpy()

            if no_bg:
                if s=="background":
                    continue
                bg = labels['background'].to_numpy()
                not_bg = np.zeros(len(bg))
                not_bg[bg==0] = 1 
                print(bg)
                print(not_bg)
                actual = np.multiply(actual, not_bg)
                pred = np.multiply(pred, not_bg)


            #finds f1 for predictions not including a specified behavior
            """
            if norm=="except":
                #define actual background
                real_frames = labels[s].to_numpy()
                pred = np.multiply(pred,real_frames)
            elif norm!="none":
                #define actual background
                bg = labels[norm].to_numpy()
                nbg = 1 - bg
                pred = np.multiply(pred,nbg)
            """

            actual = windowize(actual,window)
            pred = windowize(pred,window)


            
            #score = f1_score(actual, pred,zero_division=1)

            #calculate F1 score
            #print(s+" -> "+str(score))

            #avgScores[i] += score

            cm = confusion_matrix(actual, pred)

            print(s)
            print("f1 score"+ " " +str(f1_score(actual, pred, zero_division=1)))

            print("precision score"+ " " +str(precision_score(actual, pred, zero_division=1)))

            print("recall score"+ " " +str(recall_score(actual, pred, zero_division=1)))

            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn+fp)

            mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            nmcc = (mcc+1)/2

            print("norm mcc "+str(nmcc))

            print("specificity score"+ " " +str(specificity))

            if len(cm[0]) == 1:
                tn = cm[0][0]
                fp = 0
                fn = 0
                tp = 0

                conf_mat[i].append([tn,fp,fn,tp])
            # else:
            #     # score = 0
            #     # c = 0
            #     # window = 1200
            #     # for i in range(0, len(actual), window):
            #     #     score += f1_score(actual[i:i+window], pred[i:i+window],zero_division=1)
            #     #     c+=1

            #     #print(s + ' ' + str(score/c))

            i+=1
        
        # print()

    if False:

        boutlens = [[],[],[],[],[],[],[],[],[]]
        for f in range(0, len(fileboutlens)):
            for b in range(0,len(boutlens)):
                boutlens[b].extend(fileboutlens[f][b])
        nums = [[],[],[],[],[],[],[],[],[]]

        t = 0
        for s in behaviors:
            print(boutlens[t])
            if len(boutlens[t])!=0:
                counts, bins = np.histogram(np.array(boutlens[t]), bins=np.arange(0,1200,10))
                counts = np.multiply(np.array(counts),np.array(bins[:-1]))
                plt.clf()

                #mean = np.sum(np.multiply(np.array(counts), np.array(bins[:-1])))/np.sum(np.array(counts))
                mean = 0
                cs = []
                for c in range(0,len(counts)):
                    cs.extend([bins[c]]*counts[c])
                #print(cs)


                lower = np.percentile(cs, 25)
                median = np.percentile(cs, 50)
                higher = np.percentile(cs, 75)   
                nums[t].extend([lower, median, higher])             
                #counts = counts/np.sum(counts)
                plt.axvspan(lower, higher, color='black', alpha=0.2)
                plt.stairs(counts, bins,fill=True)
                plt.axvline(median, color='yellow')
                plt.axvline(10, color='black')

                plt.ylabel('total frame counts')
                plt.xlabel('bout length') 
                plt.title('most probable bout length of a given '+s+' frame')
                plt.savefig(os.path.join(os.getcwd(), 'bouts/'+s+".jpg"))


            t+=1
        invert = [[],[],[]]
        for n in nums:
            if len(n)==0:
                invert[0].append(0)
                invert[1].append(0)
                invert[2].append(0)
            else:
                invert[0].append(n[0])
                invert[1].append(n[1])
                invert[2].append(n[2])


        df_avgbouts = pd.DataFrame(invert, columns=behaviors, index=['25th','median','75th'])
                    
        df_avgbouts.to_csv(os.path.join(os.getcwd(), 'bouts/avgbouts.csv'), index=False)


    with np.printoptions(precision=3, suppress=True):
        print(tabulate(multi_class, behaviors, tablefmt="pretty"))
        print()

    
    
    if not dircount == 0:
        i = 0

        cms = []
        for s in behaviors:

            tn = 0
            fp = 0 
            fn = 0
            tp = 0
            for m in conf_mat[i]:
                tn += m[0]
                fp += m[1]
                fn += m[2]
                tp += m[3]

            #calculate average F1 score
            avgScores[i] = avgScores[i]/dircount
            print(s+" average f1 score -> "+str(avgScores[i]))
            i+=1

            conf_matrix = [[tn,fp],[fn,tp]]
            
            print(tn)
            print(tp)
            
            print(fn)
            print(fp)
            cms.append(conf_matrix)


        matplotlib.use('Agg')
        
        print()

        i = 0
        table = []
        
        for s in behaviors:
            cm = cms[i]

            # print(s)
            print("tn - " + str(cm[0][0]))
            print("fp - " + str(cm[0][1]))
            print("fn - " + str(cm[1][0]))
            print("tp - " + str(cm[1][1]))

            cm1 = np.array(cm,copy=True)

            cm1 = [[cm[0][0], cm[0][1]],[cm[1][0], cm[1][1]]]

            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(cm1, cmap=plt.cm.Blues, alpha=0.4)
            
            for m in range(0, 2):
                for n in range(0, 2):
                    ax.text(x=n, y=m, s="{0:.3%}".format(cm1[m][n]), va='center', ha='center', size=30)

            plt.xlabel('predictions', fontsize=30)
            plt.ylabel('actuals', fontsize=30)
            plt.title(s, fontsize=40)

            print(s+' total f1 score -> ' + "{0:.3%}".format((2*precision(cm1)*recall(cm1)/(precision(cm1)+recall(cm1)))))
            print("precision " + "{0:.3%}".format((precision(cm1))))
            print("recall " + "{0:.3%}".format((recall(cm1))))
            print("specificity " + "{0:.3%}".format((specificty(cm1))))
            
            table.append([s, "{0:.3%}".format((2*precision(cm1)*recall(cm1)/(precision(cm1)+recall(cm1)))), "{0:.3%}".format((precision(cm1))), "{0:.3%}".format((recall(cm1)))])

            plt.savefig("confusion_mats/" + s + '.png')

            plt.close('all')

            i += 1

        headers = ["behavior","f1", "precision", "recall"]
        print(tabulate(table, headers, tablefmt="pretty"))
        
        return np.array(avgScores)
        
        

    return np.array(avgScores)

if __name__ == "__main__":
    #try:
    arg = str(sys.argv[1])
        
        
    behaviors = None
    model_path = os.path.split(arg)[0]
        
    with open(os.path.join(model_path,"project_config.yaml"), 'r') as stream:
        behaviors = yaml.safe_load(stream)['project']['class_names']
        
        
    f1(arg, behaviors, val_only=True)
    #except:
        #print("Please copy the directory path of the DATA folder and include it as an argument -> python confusion_matrix.py path\DATA")
    