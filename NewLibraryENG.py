#SORTING Ver2
from tqdm.notebook import tqdm
import sklearn.preprocessing as ps
from sklearn.preprocessing import StandardScaler
from random import randint
from fastdtw import fastdtw
import copy
import pymc as pm
from gettext import find
import sys, importlib
from  McsPy.McsData import RawData
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import scipy
import pywt
#from tqdm import tqdm
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import time

def mask_cuts(cut):
    before=5 #(or 50 for 5 ms : CHECK)
    after=5
    for i in range(cut.shape[0]):
        spike=cut[i]
        min_ind=np.argmin(spike)
        max_ind=np.argmax(spike)
        first=min(min_ind,max_ind)
        if first>before:
            first=first-before
        second=max(min_ind,max_ind)
        if second<(30-after):
            second=second+after
        masked_spike=np.zeros(cut.shape[1])
        masked_spike[first:second]=spike[first:second]
        cut[i]=masked_spike
    print(cut.shape)
    return cut

def find_all_spikes(data):
    pos=[]
    neg=[]
    window_size=10000 #(1000 ms)
    pbar = tqdm(total = len(data)-window_size)
    i=0
    i_bf=0
    while i <len(data)-window_size-11:
        window=data[i:i+window_size]
        neg_window=-window
        mad=scipy.stats.median_abs_deviation(window)
        thresh=4.3*mad
        pos_peaks=find_peaks(window.ravel(),height=float(thresh))
        neg_peaks=find_peaks(neg_window.ravel(),height=float(thresh))
        lp=len(pos_peaks[0])
        ln=len(neg_peaks[0])
        mp=0
        mn=0
        if len(pos_peaks[0])>0:
            mp=pos_peaks[0][lp-1]
        for j in range(lp):
            pos.append(i+pos_peaks[0][j])
        if len(neg_peaks[0])>0:
            mn=neg_peaks[0][ln-1]
        for j in range(ln):
            neg.append(i+neg_peaks[0][j])        
        i_bf=i
        if mp==0 and mn==0:
            i+=10
        else:
            i=i+max(mp,mn)+10
        delta=i-i_bf
        pbar.update(delta)
    pbar.refresh()

    firing_rate=(len(pos)+len(neg))*10000/len(data)
    print('positive spikes',len(pos),'negative spikes',len(neg),'detected spikes:', len(pos)+len(neg), 'firing rate: ',firing_rate)
    return pos,neg

def cut(pos,neg,data):
    pre = 0.0015
    post = 0.0015
    fs=10000
    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(pos)
    pos_cut= np.empty([lunghezza_indici, prima+dopo])
    lunghezza_neg=len(neg)
    neg_cut= np.empty([lunghezza_neg, prima+dopo])
    dim = data.shape[0]
    k=0
    signal_mean=abs(np.mean(data))
    #signal_std=abs(scipy.stats.median_abs_deviation(data))
    signal_std=np.std(data)
    pos_new=[]
    for i in pos:
        #verifico che la finestra non esca dal segnale
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            #std=scipy.stats.median_abs_deviation(spike)
            std=np.std(spike)
            spike_std=(spike-media)/std
            media=np.mean(spike_std)
            std=np.std(spike_std)
            #if abs(std)<=2*abs(signal_std) and abs(media)<=10*abs(signal_mean):
            if abs(std)<=3*abs(signal_std) and abs(media)<=10*abs(signal_mean):
                pos_cut[k,:] = spike_std
                pos_new.append(i)
                k += 1
    possize=k
    pos_cut=pos_cut[0:possize]
    k=0
    neg_new=[]
    for i in neg:
        #verifico che la finestra non esca dal segnale
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            #std=scipy.stats.median_abs_deviation(spike)
            std=np.std(spike)
            spike_std=(spike-media)/std
            media=np.mean(spike_std)
            std=np.std(spike_std)            
            if abs(std)<=2*abs(signal_std) and abs(media)<=10*abs(signal_mean): 
                neg_cut[k,:] = spike_std
                neg_new.append(i)
                k  += 1
    negsize=k
    neg_cut=neg_cut[0:negsize]
    print(np.isnan(pos_cut).sum(),len(pos_cut),len(pos_new),np.isnan(neg_cut).sum(),len(neg_cut),len(neg_new))
    return pos_cut,pos_new,neg_cut,neg_new


def hdbscan_clustering(cut,spike_list,len_data):
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import numpy as np
    import math

    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    #transformed=cut

    hdbscan = HDBSCAN(min_cluster_size=100, min_samples=5)
    labels = hdbscan.fit_predict(transformed)


    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        print("HDBSCAN assigned only one cluster.")
    else:
        silhouette_avg = silhouette_score(transformed, labels)
        num_clusters = len(np.unique(labels[labels != -1]))
        print("For", num_clusters,"clusters, the silhouette score is:", silhouette_avg)

    fig = plt.figure(figsize=(4, 5))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]

        # Plot the individual cluster data
        quad = math.ceil(len(unique_labels)/2)
        plt.subplot(quad, quad, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)
        plt.title(f'Cluster {cluster_label} index {i}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        std_wave = np.std(cluster_data, axis=0)
        plt.plot(mean_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    spike_list=np.array(spike_list)
    for i in unique_labels:
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        if i !=-1:
            final_data.append(ul)
        plt.subplot(quad, quad, i + 2)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i+1])}, \n firing rate: {len(temporary_data[i+1])*10000/len_data}')
    plt.subplots_adjust(hspace=2)
    plt.show()
    return final_data

def clus(cut,clustering,spike_list,data):
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN, HDBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn import metrics
    from sklearn.metrics import silhouette_score
    from scipy.stats import kurtosis
    import skfuzzy as fuzz
    import numpy as np
    import math
    n_tries=6
    len_data=len(data)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    #transformed=cut
    list_score=[]
    DB_score=[]
    best_score=[]
    if clustering=='kmeans':
        for n in range (1,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                CH=metrics.calinski_harabasz_score(transformed, labels)
                DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"), 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                list_score.append(silhouette_avg)
                DB_score.append(DB)
                best_score.append(silhouette_avg-DB)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+2
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range 2 to ", n_tries-1, ": ",top_clusters,", with a silhouette score of: ",list_score[top_clusters-2], "\u001b[0m  \n\n")
  
        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)

        #num_clusters = n
        #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        #kmeans.fit(transformed)
        #labels = kmeans.labels_
    elif clustering=='dbscan':
        dbscan = DBSCAN(eps=1.5, min_samples=60)
        labels = dbscan.fit_predict(transformed)
    elif clustering == 'fuzzy':
        for n in range (1,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                CH=metrics.calinski_harabasz_score(transformed, labels)
                DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"), 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                list_score.append(silhouette_avg)
                DB_score.append(DB)
                best_score.append(silhouette_avg-DB)
                del(u)
                del(labels)
        
        #top_clusters = list_score.index(max(list_score))+2
        #top_clusters = DB_score.index(min(DB_score))+2
        top_clusters = (best_score.index(max(best_score)))+2
        #creare vettore con (silhouette - DB) e selezionare massimo
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range 2 to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",list_score[top_clusters-2],'DB:',DB_score[top_clusters-2], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
        #num_clusters = n
        #cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, num_clusters, 2, error=0.005, maxiter=3000, init=None)
        #labels = np.argmax(u, axis=0)
    elif clustering=='hdbscan':
        hdbscan = HDBSCAN(min_cluster_size=100, min_samples=5, leaf_size=30) 
        labels = hdbscan.fit_predict(transformed)

    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    print(unique_labels)
    
    if len(unique_labels) == 1:
        print("DBSCAN assigned only one cluster.")
    else:
        silhouette_avg = silhouette_score(transformed, labels)
        num_clusters = len(np.unique(labels[labels != -1]))
        print("For", top_clusters,"clusters, the silhouette score is:", list_score[top_clusters-2])

    fig = plt.figure(figsize=(8, 10))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        #final_data.append(spike_list[labels == cluster_label].tolist())

        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=len(unique_labels)
            size2=1
        else:
            size1 = math.ceil(len(unique_labels)/2)
            size2=size1
        plt.subplot(size1,size2, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        std_wave = np.std(cluster_data, axis=0)
        plt.plot(mean_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')

    # Adjust layout to prevent overlapping
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    spike_list=np.array(spike_list)
    for i in range(0,len(unique_labels)):
        print(i)
        #if clustering=='dbscan' or 'hdbscan':
        #    k=i+1
        #else:
        #    k=i
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>2:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i])}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
        #k+=1
    plt.subplots_adjust(hspace=2)
    plt.show()
    del(unique_labels)
    return final_data
#################

#____________________________________________________________________________________FILT BUTTERWORTH_________________________________

from scipy.signal import ellip, cheby1, bessel, butter, lfilter, filtfilt, iirfilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Implementation of Butterworth filtering.
    
    Params:
     - lowcut: low cutting frequency (Hz)
     - highcut: high cutting frequency (Hz)
     - fs: sampling frequency (Hz)
     - order: order of the filter
    
    Return:
     - b,a = coefficients of the filter
    
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Perfom the filtering of the data using a zero-phase Butterworth filter.
    
    Params:
     - data: The signal as a 1-dimensional numpy array
     - lowcut: low cutting frequency (Hz)
     - highcut: high cutting frequency (Hz)
     - fs: sampling frequency (Hz)
     - order: order of the filter
     
    Return:
     - y = signal filtered
    
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y