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
    #print(cut.shape)
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
            std=np.std(spike)
            spike_std=(spike-media)/std
            media=np.mean(spike_std)
            std=np.std(spike_std)
            if abs(std)<=2*abs(signal_std) and abs(media)<=10*abs(signal_mean):
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
    #print(np.isnan(pos_cut).sum(),len(pos_cut),len(pos_new),np.isnan(neg_cut).sum(),len(neg_cut),len(neg_new))
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

def clus(cut,clustering,spike_list,data,flag=0):
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
    n_min=2
    n_tries=11
    spike_list=np.array(spike_list)
    #print('len spike list: ',len(spike_list))
    #print('n tries: ', int(len(spike_list)/1000))
    #n_tries=int(len(spike_list)/1000)+1
    len_data=len(data)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    #list_score=[]
    #DB_score=[]
    best_score=[]
    if clustering=='kmeans':
        for n in range (n_min,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+n_min
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range", n_min, "to ", n_tries-1, ": ",top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min], "\u001b[0m  \n\n")
  
        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)

    elif clustering == 'fuzzy':
        for n in range (n_min,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(2*silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(u)
                del(labels)
        
        top_clusters = (best_score.index(max(best_score)))+n_min
        #creare vettore con (silhouette - DB) e selezionare massimo
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range",n_min," to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min])#,'DB:',DB_score[top_clusters-n_min], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
    
    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    firings=np.zeros(len(unique_labels))
    
    fig = plt.figure(figsize=(10, 12))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        firings[i]=len(cluster_data)*10000/len_data
        
        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=len(unique_labels)
            size2=1
        elif len(unique_labels)<=5:
            size1 = math.ceil(len(unique_labels)/2)
            size2=size1
        elif len(unique_labels)<=8:
            size1 = 3
            size2=math.ceil(len(unique_labels)/size1)
        else:
            size1=6
            size2=math.ceil(len(unique_labels)/size1)
        plt.subplot(size1,size2, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        info.append(f'mean clus{cluster_label}')
        info.append(mean_wave)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
    plt.tight_layout()

    mean_firing=np.mean(firings)
    std_firing=np.std(firings)
    firing_threshold=mean_firing-3*std_firing
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    for i in range(0,len(unique_labels)):
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>firing_threshold:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i])}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
    plt.subplots_adjust(hspace=2.5)
    plt.show()

    del(unique_labels)
    return final_data
#################

def nested_clus(cut,clustering,spike_list,data,flag=0):
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
    n_min=2
    n_tries=11
    spike_list=np.array(spike_list)
    #print('len spike list: ',len(spike_list))
    #print('n tries: ', int(len(spike_list)/1000))
    #n_tries=int(len(spike_list)/1000)+1
    len_data=len(data)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    #list_score=[]
    #DB_score=[]
    best_score=[]
    if clustering=='kmeans':
        for n in range (n_min,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+n_min
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range", n_min, "to ", n_tries-1, ": ",top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min], "\u001b[0m  \n\n")
  
        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)

    elif clustering == 'fuzzy':
        for n in range (n_min,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(2*silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(u)
                del(labels) 
            if silhouette_avg<0.08:
                break       
        top_clusters = (best_score.index(max(best_score)))+n_min
        #creare vettore con (silhouette - DB) e selezionare massimo
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range",n_min," to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min])#,'DB:',DB_score[top_clusters-n_min], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
    
    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    firings=np.zeros(len(unique_labels))
    
    fig = plt.figure(figsize=(10, 12))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        firings[i]=len(cluster_data)*10000/len_data
        
        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=len(unique_labels)
            size2=1
        elif len(unique_labels)<=5:
            size1 = math.ceil(len(unique_labels)/2)
            size2=size1
        elif len(unique_labels)<=8:
            size1 = 3
            size2=math.ceil(len(unique_labels)/size1)
        else:
            size1=6
            size2=math.ceil(len(unique_labels)/size1)
        plt.subplot(size1,size2, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        info.append(f'mean clus{cluster_label}')
        info.append(mean_wave)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
    plt.tight_layout()

    mean_firing=np.mean(firings)
    std_firing=np.std(firings)
    firing_threshold=mean_firing-3*std_firing
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    for i in range(0,len(unique_labels)):
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>firing_threshold:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i])}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
    plt.subplots_adjust(hspace=2.5)
    plt.show()
    if top_clusters==2 and flag==0:
        subgroup_indices = final_data[0]
        spike_list0 = [np.where(np.isin(spike_list, indices))[0] for indices in subgroup_indices]
        cut0 = [cut[pos] for pos in spike_list0]
        cut_np = np.array(cut0)
        cut0 = cut_np.reshape(cut_np.shape[0], -1)
        spike_list0 = np.concatenate([arr.flatten() for arr in spike_list0])
        subgroup_indices = final_data[1]
        spike_list1 = [np.where(np.isin(spike_list, indices))[0] for indices in subgroup_indices]
        cut1 = [cut[pos] for pos in spike_list1]
        cut_np = np.array(cut1)
        cut1 = cut_np.reshape(cut_np.shape[0], -1)
        spike_list1 = np.concatenate([arr.flatten() for arr in spike_list1])
        
        print(len(cut0),len(cut1))
        if len(cut0)<=7000 | len(cut1)<=7000:
            flag=1
        print('first sub-clustering')
        final_data=clus(cut0,'kmeans',spike_list0,data)
        print('second sub-clustering')
        final_data.append(clus(cut1,'kmeans',spike_list1,data))

    else:
        del(unique_labels)
        return final_data
#################


def switch_clus(cut,clustering,spike_list,data,switch_index):
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
    n_min=3
    n_tries=15
    spike_list=np.array(spike_list)
    #print('len spike list: ',len(spike_list))
    #print('n tries: ', int(len(spike_list)/1000))
    #n_tries=int(len(spike_list)/1000)+1
    len_data=len(data)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    #list_score=[]
    #DB_score=[]
    best_score=[]
    if clustering=='kmeans':
        for n in range (n_min,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+n_min
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range", n_min, "to ", n_tries-1, ": ",top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min], "\u001b[0m  \n\n")
  
        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)

    elif clustering == 'fuzzy':
        for n in range (n_min,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(2*silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(u)
                del(labels)
        
        top_clusters = (best_score.index(max(best_score)))+n_min
        #creare vettore con (silhouette - DB) e selezionare massimo
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range",n_min," to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min])#,'DB:',DB_score[top_clusters-n_min], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
    
    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    firings=np.zeros(len(unique_labels))
    
    fig = plt.figure(figsize=(10, 12))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        firings[i]=len(cluster_data)*10000/len_data
        
        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=len(unique_labels)
            size2=1
        elif len(unique_labels)<=5:
            size1 = math.ceil(len(unique_labels)/2)
            size2=size1
        elif len(unique_labels)<=8:
            size1 = 3
            size2=math.ceil(len(unique_labels)/size1)
        else:
            size1=6
            size2=math.ceil(len(unique_labels)/size1)
        plt.subplot(size1,size2, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        info.append(f'mean clus{cluster_label}')
        info.append(mean_wave)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
    plt.tight_layout()

    mean_firing=np.mean(firings)
    std_firing=np.std(firings)
    firing_threshold=mean_firing-std_firing
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    for i in range(0,len(unique_labels)):
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>firing_threshold:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i])}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
    plt.subplots_adjust(hspace=2.5)
    plt.show()
    del(unique_labels)
    return final_data
##################
def bounded_clus(cut,clustering,spike_list,data):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn import metrics
    from sklearn.metrics import silhouette_score
    from scipy.stats import kurtosis
    import skfuzzy as fuzz
    import numpy as np
    import math
    spike_list=np.array(spike_list)
    len_data=len(data)
    n_min=int((10000*len(spike_list))/(2*len_data))
    n_tries=int(((10000*len(spike_list))/(0.16*len_data)+1))
    print('cluster range: ',n_min,n_tries)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    list_score=[]
    DB_score=[]
    best_score=[]
    #CH_score=[]
    if clustering=='kmeans':
        for n in tqdm(range (n_min,n_tries)):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                DB=metrics.davies_bouldin_score(transformed, labels)
                #print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"),'DB score',format(DB, ".3f"))
                list_score.append(silhouette_avg)
                DB_score.append(DB/10)
                #best_score.append(silhouette_avg-(DB/10))
                #CH_score.append(CH/3000)
                #best_score.append(silhouette_avg)
                del(model)
                del(labels)
        #top_clusters = (list_score.index(max(list_score)))+n_min
        #top_DB=(DB_score.index(max(DB_score)))+n_min
        top_clusters=(DB_score.index(min(DB_score)))+n_min
        print('top cluster: ',(DB_score.index(min(DB_score))),'+',n_min,'=',top_clusters)
        #n_clusters=top_clusters
        plt_value=min(DB_score)
        plt_index=DB_score.index(min(DB_score))
        '''
        if top_clusters==top_DB:
            num_clusters=top_clusters
        else:
            num_clusters=(DB_score.index(max(DB_score)))+n_min
        '''    
        plt.figure()
        #x=np.arange(n_min,n_tries)
        plt.plot(list_score)
        plt.plot(DB_score)
        #plt.plot(CH_score)
        plt.title('Silhouette score')
        plt.xlabel('Number of clusters')
        #plt.xlabel('x')
        #plt.ylim(0,2)
        plt.scatter((list_score.index(max(list_score))), max(list_score), c='red', marker='o', label='Maximum s score')
        plt.scatter(plt_index, plt_value, c='green', marker='o', label='Minimum DB score')
        #plt.scatter((CH_score.index(max(CH_score))), max(CH_score), c='blue', marker='o', label='Maximum CH score')
        #plt.scatter((best_score.index(min(best_score))), min(best_score), c='blue', marker='x', label='Best sil - DB score')
        plt.show()

        #top_clusters=num_clusters
        print("\n\n\033[1;31;47mBest cluster in the range", n_min, "to ", n_tries-1, ": ",plt_index,top_clusters,", with a silhouette score of: ",list_score[top_clusters-n_min],'DB score: ',DB_score[top_clusters-n_min],plt_value, "\u001b[0m  \n\n")

        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)
    elif clustering == 'fuzzy':
        for n in range (n_min,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                list_score.append(silhouette_avg)
                DB_score.append(DB)
                #best_score.append(2*silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(u)
                del(labels)

        top_clusters = (best_score.index(max(best_score)))+n_min
        plt.figure()
        #x=np.arange(n_min,n_tries)
        plt.plot(list_score)
        plt.plot(DB_score)
        #plt.plot(CH_score)
        plt.title('Silhouette score')
        plt.xlabel('Number of clusters')
        #plt.xlabel('x')
        #plt.ylim(0,2)
        plt.scatter((list_score.index(max(list_score))), max(list_score), c='red', marker='o', label='Maximum s score')
        plt.scatter(plt_index, plt_value, c='green', marker='o', label='Minimum DB score')
        #plt.scatter((CH_score.index(max(CH_score))), max(CH_score), c='blue', marker='o', label='Maximum CH score')
        #plt.scatter((best_score.index(min(best_score))), min(best_score), c='blue', marker='x', label='Best sil - DB score')
        plt.show()
        #creare vettore con (silhouette - DB) e selezionare massimo
        print("\n\n\033[1;31;47mBest cluster in the range",n_min," to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",best_score[top_clusters-n_min])#,'DB:',DB_score[top_clusters-n_min], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
        

    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    firings=np.zeros(len(unique_labels))
    fig = plt.figure(figsize=(10, 12))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        firings[i]=len(cluster_data)*10000/len_data

        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=len(unique_labels)
            size2=1
        elif len(unique_labels)<=5:
            size1 = math.ceil(len(unique_labels)/2)
            size2=size1
        elif len(unique_labels)<=8:
            size1 = 3
            size2=math.ceil(len(unique_labels)/size1)
        else:
            size1=6
            size2=math.ceil(len(unique_labels)/size1)
        plt.subplot(size1,size2, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        info.append(f'mean clus{cluster_label}')
        info.append(mean_wave)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
    plt.tight_layout()

    mean_firing=np.mean(firings)
    std_firing=np.std(firings)
    firing_threshold=mean_firing-std_firing
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    for i in range(0,len(unique_labels)):
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>firing_threshold:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
    plt.subplots_adjust(hspace=2.5)
    plt.show()
    del(unique_labels)
    return final_data
###############

def NEWclus(cut,clustering,spike_list,data):
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
    n_tries=15
    len_data=len(data)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    #list_score=[]
    #DB_score=[]
    best_score=[]
    if clustering=='kmeans':
        for n in range (1,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                #best_score.append(silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+2
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range 2 to ", n_tries-1, ": ",top_clusters,", with a silhouette score of: ",list_score[top_clusters-2], "\u001b[0m  \n\n")
  
        model = KMeans(n_clusters=top_clusters, n_init='auto', copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)

    elif clustering == 'fuzzy':
        for n in range (1,n_tries):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, n, 2, error=0.005, maxiter=3000, init=None)
            labels = np.argmax(u, axis=0)
            if (n !=1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                #DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"))#, 'CH score',format(CH, ".3f"),'DB score',format(DB, ".3f"))
                #list_score.append(silhouette_avg)
                #DB_score.append(DB)
                best_score.append(silhouette_avg)#-DB)
                del(u)
                del(labels)
        
        top_clusters = (best_score.index(max(best_score)))+2
        #creare vettore con (silhouette - DB) e selezionare massimo
        num_clusters=top_clusters
        print("\n\n\033[1;31;47mBest cluster in the range 2 to ", n_tries-1, ":" ,top_clusters,", with a silhouette score of: ",list_score[top_clusters-2],'DB:',DB_score[top_clusters-2], "\u001b[0m  \n\n")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(transformed.T, top_clusters, 2, error=0.005, maxiter=3000, init=None)
        labels = np.argmax(u, axis=0)
    
    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    clusters=clustersdtw(cut,labels,unique_labels)
    clus=[]
    cluster_data=[]
    firings=np.zeros(len(clusters))
    for i in range(len(clusters)):
        clus.append(i)
        clusterings=[]
        for j in range(len(clusters[i])):
            clusterings.extend(cut[labels==clusters[i][j]])
        cluster_data.append(clusterings)
        arr_cd=np.array(cluster_data)
        firings[i]=len(arr_cd)*10000/len_data
        # Plot the individual cluster data
        if len(clus)<=2:
            size1=len(clus)
            size2=1
        elif len(clus)<=5:
            size1 = math.ceil(len(clus)/2)
            size2=size1
        elif len(clus)<=8:
            size1 = 3
            size2=math.ceil(len(clus)/size1)
        else:
            size1=6
            size2=math.ceil(len(clus)/size1)
        plt.subplot(size1,size2, i + 1)
        plt.plot(np.array(arr_cd).transpose(), alpha=0.5)  # Use alpha for transparency
        plt.title(f'Cluster {i} \n numerosity: {len(arr_cd)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        info.append(f'mean clus{cluster_label}')
        info.append(mean_wave)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
    plt.tight_layout()

    mean_firing=np.mean(firings)
    std_firing=np.std(firings)
    firing_threshold=mean_firing-std_firing
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    spike_list=np.array(spike_list)
    for i in range(0,len(clus)):
        ul=spike_list[labels==i]
        temporary_data.append(ul)
        fr=len(temporary_data[i])*10000/len_data
        if i != -1 and fr>firing_threshold:
            final_data.append(ul)
        plt.subplot(size1, size2, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i])}, \n firing rate: {format(len(temporary_data[i])*10000/len_data, ".3f")}')
    plt.subplots_adjust(hspace=2.5)
    plt.show()
    del(unique_labels)
    return final_data, info


#########################
def clustersdtw(cut,labels,unique_labels):
    u_l=copy.deepcopy(unique_labels)
    labs=unique_labels
    clusters=[]
    list0=[]
    list0.append(0)
    clusters.append(list0)
    for i, first_cluster_loop in enumerate(u_l):
        spike1=np.mean(cut[labels==first_cluster_loop],axis=0)
        check=0
        for i in range(len(clusters)):
            if first_cluster_loop in clusters[i]:
                check=1
        if check==0:
            list0=[]
            list0.append(first_cluster_loop)
            clusters.append(list0)
        for j, second_cluster_loop in enumerate(u_l[i+1:]):
            spike2=np.mean(cut[labels==second_cluster_loop],axis=0)
            distance,path=fastdtw(spike1, spike2)
            if distance<=4 and first_cluster_loop!=second_cluster_loop:
                for arr in clusters:
                    if first_cluster_loop in arr and second_cluster_loop not in arr:
                        arr.append(second_cluster_loop)
                        ind=np.where(labs==second_cluster_loop)
                        labs=np.delete(labs,ind)
                    elif first_cluster_loop not in arr and second_cluster_loop in arr:
                        arr.append(first_cluster_loop)
                        ind=np.where(labs==first_cluster_loop)
                        labs=np.delete(labs,ind)                    

    return clusters

################################POINT PROCESS
def Bayesian_mixture_model(ISI_data):
    with pm.Model() as model:
        ##### WALD DISTRIBUTION (INVERSE GAUSSIAN)
        mu1 = pm.Uniform('mu1',lower=0.01,upper=0.1)
        lam1 = pm.Uniform('lam1',lower=0.01,upper=0.04)
        obs1 = pm.Wald.dist(mu=mu1,lam=lam1)


        mu2 = pm.Uniform('mu2',lower=0,upper=0.2)
        sigma2 = pm.Uniform('sigma2',lower=0.0001,upper=0.5)
        obs2 = pm.TruncatedNormal.dist(mu=mu2, sigma=sigma2, lower=0.0)

        mu3 = pm.Uniform('mu3',lower=0.1,upper=0.6)
        sigma3 = pm.Uniform('sigma3',lower=0.0001,upper=0.5)
        obs3 = pm.TruncatedNormal.dist(mu=mu3, sigma=sigma3, lower=0.0)


        w = pm.Dirichlet('w', a=np.array([1., .4, .4]))
        #w = pm.Dirichlet('w', a=np.array([1., .4]))

        like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2, obs3], observed=ISI_data)
        #like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2], observed=ISI_data)

        step = pm.NUTS(target_accept=0.9)
        trace = pm.sample(step=step,draws=1000,chains=1,tune=1000,cores=4)
        
        ppc_trace = pm.sample_posterior_predictive(trace,model=model)
        
    map_estimate = pm.find_MAP(model=model)
    
    del map_estimate['w_simplex__']
    del map_estimate['mu1_interval__']
    del map_estimate['lam1_interval__']
    del map_estimate['mu2_interval__']
    del map_estimate['sigma2_interval__']
    del map_estimate['mu3_interval__']
    del map_estimate['sigma3_interval__']
    
    map_estimate['w1'] = map_estimate['w'][0]
    map_estimate['w2'] = map_estimate['w'][1]
    map_estimate['w3'] = map_estimate['w'][2]

    del map_estimate['w']


    return map_estimate, ppc_trace
#######################




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