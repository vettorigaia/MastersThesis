#SORTING Ver2
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
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import time

def find_all_spikes_new(data):
    pos=[]
    neg=[]
    neg_data=-data
    mad=scipy.stats.median_abs_deviation(data)
    thresh=3*mad
    pos_peaks=find_peaks(data.ravel(),height=float(thresh))
    neg_peaks=find_peaks(neg_data.ravel(),height=float(thresh))
    pos=pos_peaks[0]
    neg=neg_peaks[0]
    firing_rate=(len(pos)+len(neg))*10000/len(data)
    print('detected spikes:', len(pos)+len(neg), 'firing rate: ',firing_rate)
    return pos,neg

def find_all_spikes(data):
    pos=[]
    neg=[]
    window_size=300 #(30 ms)
    pbar = tqdm(total = len(data)-window_size)
    i=0
    i_bf=0
    while i <len(data)-window_size-11:
        window=data[i:i+window_size]
        neg_window=-window
        mad=scipy.stats.median_abs_deviation(window)
        thresh=4*mad
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

    firing_rate=(len(pos)+len(neg))*10000/len(data)
    print('detected spikes:', len(pos)+len(neg), 'firing rate: ',firing_rate)
    return pos,neg

def cut(pos,neg,data):
    pre = 0.003
    post = 0.003
    fs=10000
    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(pos)
    pos_cut= np.empty([lunghezza_indici+1, prima+dopo])
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

def RMM(data):
    # window size 30ms, threshold for first spike: 3*mad(window), threshold for second spike: 1.1 mean(window)
    # differential threshold: 6*mad(window)
    window_size=300 #0.03 sec (30ms)
    i=0
    first_peaks_set=set()
    second_peaks_set=set()
    abs_data=abs(data)
    pbar = tqdm(total = len(data)-window_size)

   # with tqdm(total=100) as pbar:
    while i<=len(data)-window_size-50:

        i_bf=i
        entrato=False
        found=False
        abs_window=abs_data[i:i+window_size]
        window=data[i:i+window_size]
        mad=scipy.stats.median_abs_deviation(window)
        thresh=3*mad
        media=1.1*np.mean(window)
        first_peaks=find_peaks(abs_window.ravel(),height=float(thresh))
        len1=len(first_peaks[0])
        second_peaks=find_peaks(abs_window.ravel(),height=float(media))
        len2=len(second_peaks[0])
        if len1==0 or len2==0:
            i=i+1
        else:
            if entrato==False:
                entrato=True
                for k in range(len(first_peaks[0])):
                    for j in range(len(second_peaks[0])):
                        if second_peaks[0][j]>first_peaks[0][k]:
                            #cioè se il secondo picco è successivo e se il primo picco ha un valore maggiore in assoluto
                            primo=abs_window[first_peaks[0][k]]
                            secondo=abs_window[second_peaks[0][j]]
                            #diff=abs_window[first_peaks[0][k]] + abs_window[second_peaks[0][j]]
                            diff=primo+secondo

                            if found==False and window[first_peaks[0][k]]*window[second_peaks[0][j]]<0 and diff  > 2*thresh and primo>1.5*secondo:# and abs_window[first_peaks[0][j]]>abs_window[second_peaks[0][k]] and abs_window[second_peaks[0][j]]<3*media:
                                #cioè se non sono stati trovati già due indici che soddisfano, se i picchi hanno segni opposti e hanno una distanza sopra la soglia
                                peaks_indices=(i+first_peaks[0][k],i+second_peaks[0][j])
                                found=True
                                #print('found',i)
                                first_peaks_set.add(peaks_indices[0])
                                second_peaks_set.add(peaks_indices[1])
                                i=peaks_indices[1]+10
                first_peaks=[]
                second_peaks=[]
                i=i+1
            else:
                i=i+1
        #time.sleep(0.1)  # Adjust the sleep duration as needed
        delta=i-i_bf
        pbar.update(delta)
    minima=sorted(list(first_peaks_set))
    maxima=sorted(list(second_peaks_set))
    firing_rate=len(minima)*10000/len(data)
    print('detected spikes:', len(minima), 'firing rate: ',firing_rate)
    return minima, maxima



def find_spikes(data):
    window_size = 300  # (100 samples 0.01 sec)
    research_range = 100  # check for a positive peak in the next 0.02 s (200 samples)
    i = 0
    local_minima_indices = set()
    local_maxima_indices = set()
    abs_data=abs(data)
    pbar = tqdm(total = len(data)-window_size-10)
    while i <= len(data) - window_size -10:
        i += 10
        i_bf=i
        abs_window = abs_data[i:i + window_size]
        window = data[i:i + window_size]
        min_index = np.argmax(abs_window)
        local_minima_index = i + min_index
        local_min = data[local_minima_index]

        mad = scipy.stats.median_abs_deviation(window)
        previous_thresh = 3 * mad

        if abs(local_min) >= previous_thresh:
            research_zero = local_minima_index + 1
            research_window = data[research_zero:research_zero + research_range]
            media = np.mean(research_window)
            signal = research_window

            if local_min > 0:
                signal = -research_window

            max_peaks = find_peaks(signal.ravel(), height=float(media))
            d_thresh = 7 * scipy.stats.median_abs_deviation(research_window)
            selected_indices = []

            for max_peak in max_peaks[0]:
                value = signal[max_peak]
                primo=abs(local_min)
                secondo=abs(value)
                diff=primo+secondo

                if (value < previous_thresh) and (diff > d_thresh) and primo>1.3*secondo:
                    selected_indices.append(max_peak)

            if not selected_indices:
                i += 50
            else:
                chosen_max = min(selected_indices)
                local_max_index = research_zero + chosen_max

                local_minima_indices.add(local_minima_index)
                
                local_maxima_indices.add(local_max_index)

                i = local_max_index + 10
        else:
            i += 5
        delta=i-i_bf
        pbar.update(delta)

    unique_minima_indices = list(local_minima_indices)
    unique_maxima_indices = list(local_maxima_indices)
    firing_rate=len(unique_minima_indices)*10000/len(data)
    print('detected spikes:', len(unique_minima_indices),'firing rate: ',firing_rate)

    return unique_minima_indices, unique_maxima_indices


def find_spikes_with_memory(data):
    window_size = 300  # (100 samples 0.01 sec)
    research_range = 100  # check for a positive peak in the next 0.02 ms (200 samples)
    i = 0
    local_minima_indices = set()
    local_maxima_indices = set()
    last_50_peaks = []
    last_50_diff = []
    abs_data=abs(data)
    pbar = tqdm(total = len(data)-window_size-10)
    while i <= len(data) - window_size - 10 + 1:
        i += 10
        i_bf=i
        abs_window = abs_data[i:i + window_size]
        window = data[i:i + window_size]
        min_index = np.argmax(abs_window)
        local_minima_index = i + min_index
        local_min = data[local_minima_index]
        if i <=5000:
            mad = scipy.stats.median_abs_deviation(window)
            previous_thresh = 4 * mad
        else:
            previous_thresh=0.85*uno

        if abs(local_min) >= previous_thresh:
            research_zero = local_minima_index + 1
            research_window = data[research_zero:research_zero + research_range]
            media = np.mean(research_window)
            signal = research_window

            if local_min > 0:
                signal = -research_window

            max_peaks = find_peaks(signal.ravel(), height=float(media))
            if i<=5000:
                d_thresh = 7 * scipy.stats.median_abs_deviation(research_window)
            else:
                d_thresh=0.85*due
            selected_indices = []

            for max_peak in max_peaks[0]:
                value = signal[max_peak]
                diff=abs(value) +abs(local_min)

                if (value < previous_thresh+1) and (diff > d_thresh):
                    selected_indices.append(max_peak)

            if not selected_indices:
                i += 50
            else:
                chosen_max = min(selected_indices)
                local_max_index = research_zero + chosen_max
                local_max = data[local_max_index]

                local_minima_indices.add(local_minima_index)
                last_50_peaks.append(abs(local_min))
                last_50_diff.append(abs(local_min - local_max))
                uno = np.mean(last_50_peaks)
                due = np.mean(last_50_diff)
                local_maxima_indices.add(local_max_index)

                if len(last_50_peaks) > 5000:
                    last_50_peaks.pop(0)

                if len(last_50_diff) > 5000:
                    last_50_diff.pop(0)

                i = local_max_index + 50
        else:
            i += 10
        delta=i-i_bf
        pbar.update(delta)

    unique_minima_indices = list(local_minima_indices)
    unique_maxima_indices = list(local_maxima_indices)
    firing_rate=len(unique_minima_indices)*10000/len(data)
    print('detected spikes:', len(unique_minima_indices),'firing rate: ',firing_rate)
    
    return unique_minima_indices, unique_maxima_indices


def DetectSpike(segnale, soglia, fs, dead_time = 0.003):   
 
    """
    Detect spikes when the signal exceed in amplitude a certain threshold
    It works automatically with both positive and negativa thresholds
    
    PARAMETERS
    segnale = the signal in which it search for the peaks (it must be a numpy.array or a list or a tuple)
    soglia = the chosen thresold for the channel(it has to be int, float, double, numpy.float64)
    fs = sampling frequency (it must be an integer)
    dead_time [optional] = time in which the function doesn't search for a maximum after detecting one
    
    RETURN
    indici_spike[:k] = python list with length m, which contains all the samples of the signal when it cross the threshold
    
    """
    
    j = 0 
    numero_campioni = 0
    for j in segnale:
        numero_campioni += 1
    soglia = abs(soglia) #we consider both a positive and negative thresholds
    indici_spike = [None] * numero_campioni
    dead_campioni = int(dead_time*fs)
    i = 0
    k = 0
    while (i<numero_campioni):
        valore = abs(segnale[i])
        if (valore>soglia):
            indici_spike[k] = i
            k += 1
            i += dead_campioni
        else:
            i += 1
    return indici_spike[:k] 



def AlignSpike(segnale, indici, soglia, fs, research_time = 0.002):  
    """
    Align all the spikes previously detected with the function RilevaSpike
    
    PARAMETERS
    segnale = the signal to search the peaks in (it must be a numpy.array, or a list or a tuple)
    indici = the samples detected with the function RilevaSpike (they must be type integer)
    fs = sampling frequency (it must be type integer)
    research_timpe [optional] = time (in seconds) in which the function search for the relative maximum (default 0.002)
    
    RETURN
    indici_spike[:k] = a python list of length m, which contains all the samples of the spikes aligned to the minimum or maximum (if the signal excedees both, they are aligned to the minimum)
     
    """

    numero_campioni = len(segnale)
    research_campioni = int(research_time*fs)
    indici_allineati = [None] * numero_campioni
    soglia = abs(soglia)

    m=0
    for i in indici:
        k = 0
        picco_negativo = False  
        if (i + research_campioni) <= numero_campioni:
            while (k<research_campioni):   
                if segnale[i+k] < -soglia:
                    picco_negativo = True
                    indici_allineati[m] = i+k 
                    k+=1
                    break
                k+=1
            if picco_negativo == False:
                indici_allineati[m] = i 
                k=0 
                while (k<research_campioni):
                    if segnale[i+k] > segnale[indici_allineati[m]]:
                        indici_allineati[m] = i+k
                    k += 1     
            else:
                while (k<research_campioni):
                    if segnale[i+k] < segnale[indici_allineati[m]]:
                        indici_allineati[m] = i+k
                    k += 1
            m +=1    
        else:
            break
    return indici_allineati[:m]



def ExtractSpike(segnale, indici, fs, pre = 0.001, post = 0.002):
    """
    Extract the waveform of the spikes as an array
    
    PARAMETERS:
    segnale: the signal as an unidimensional numpy array 
    indice: the samples of the spikes, as a unidimensional numpy array
    pre: length of the cutoff in seconds before the spike
    post: length of the cutoff in seconds after the spike
    fs: sampling frequency
    
    RETURNS
    cutouts: bidimensional numpy array, with a spike in each row

    """

    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(indici)
    cutout = np.empty([lunghezza_indici, prima+dopo], np.int32)
 
    dim = segnale.shape[0]
    k=0
    for i in indici:
        #verifico che la finestra non esca dal segnale
        if (i-prima >= 0) and (i+dopo <= dim):
            cutout[k] = segnale[(int(i)-prima):(int(i)+dopo)]          
        k += 1
    return cutout


#____________________________________________________________________________________________PCA_________________________________________________


def EseguiPCA(dati, n=3, show=False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    #Standardizing data
    standardizzati = StandardScaler().fit_transform(dati)
    print("\nSignal standardized\nMean: ", np.mean(standardizzati), "\nVariance: ", np.std(standardizzati)**2, "\n")

    #3D
    if n==3:
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(standardizzati)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2','PC3'])

        if show == True:
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(principal_components[:,0], principal_components[:,1], principal_components[:,2], color='#000000', depthshade=True, lw=0)  
            plt.show()
        
    #2D    
    elif n==2:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(standardizzati)
        principal_DataFrame = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

        if show == True:
            fig = plt.figure(figsize = (15,15))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('Principal Component 1', fontsize = 30)
            ax.set_ylabel('Principal Component 2', fontsize = 30)

            x = principal_components[:,0]
            y = principal_components[:,1]
            ax.scatter(x, y, color ="#000000")
            ax.grid()
            plt.show()
    
    else:
        raise Exception("PCA funciton only work with 2 or 3 dimensions! n=2, n=3")
    
    return principal_DataFrame


#____________________________________________________________________________________________HIERARCHICAL_________________________________________________
def funzionedscan(cutouts,transformed,cluster_labels,minima):
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    import sklearn.preprocessing as ps

    spike_list=np.array(minima)
    list_score = [] 
    #spike_list=np.array(aligned_indexes[legend[electrode].values[0]])
    out=0
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
        
    for i in range(cluster_labels.shape[0]):
        if (cluster_labels[i] == -1):
            out+=1
        elif (cluster_labels[i] == 0):
            a +=1
        elif (cluster_labels[i] == 1):
            b+=1        
        elif (cluster_labels[i] == 2):
            c+=1
        elif (cluster_labels[i] == 3):
            d+=1
        elif (cluster_labels[i] == 4):
            e+=1
        elif (cluster_labels[i] == 5):
            f+=1

    if(out!=0):
        print('\nSpike detected as noise', out)
    else:
        print('\nNo spike detected as noise')

    check=0

    color = []
    for i in cluster_labels:
        color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        if i==-1:
            color[i]='k'

    indici=cluster_labels
    un=unique(cluster_labels)
    print('cluster_labels',un)
    coordinate=transformed  

    ###NUOVO
    indices_to_remove = []  # Crea una lista per tenere traccia degli indici da rimuovere


    #############perché controllo su b e non su a?
    if(a!=0):
        for i in reversed(range(indici.shape[0])):
            if (indici[i] == -1):
                #### nuovo
                print('removing')
                indices_to_remove.append(i)
                print(i)
                #print(coordinate[i])
                # vecchio np.delete(coordinate, i)
                # vecchio np.delete(indici, i)
            ###NUOVO (elimino fuori dal ciclo così la modifica è permanente
            coordinate = np.delete(coordinate, indices_to_remove, axis=0)
            indici = np.delete(indici, indices_to_remove)
            silhouette_avg = silhouette_score(coordinate, indici)
            print("\nNumber of clusters: ", len(set(cluster_labels))-1,"\nThe silhouette score is:", silhouette_avg)
            list_score.append(silhouette_avg)
            check=1
            break

    if check==0:
        print('\nOnly one cluster detected')
    if(a!=0):
        print('\nBlue spikes:', a)
    if(b!=0):
        print('\nOrange spikes:', b)
    if(c!=0):
        print('\nGreen spikes:', c)
    if(d!=0):
        print('\nRed spikes:', d)
    if(e!=0):
        print('\nPurple spikes:', e)
    if(f!=0):
        print('\nBrown spikes:', f)

    #Plot PCA 
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')

    #Plot average waveform
    ax = fig.add_subplot(1, 2, 2)
    for i in range(len(set(cluster_labels))-1):
        idx = cluster_labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        mean_wave = np.mean(cutouts[idx,:],axis = 0)
        std_wave = np.std(cutouts[idx,:],axis = 0)
        ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)

    plt.xlabel('Time [0.1ms]')
    plt.ylabel('Voltage [\u03BCV]')
    plt.show()    

    list_idx = list(np.unique(cluster_labels))
    final_list = []

    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]

    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])       

    return final_list

#____________________________________________________________________________________________HIERARCHICAL_________________________________________________

def various_clustering(clustering,n,cutouts,spike_list,fs,n_comp=3,centroids=False):
    import sys,os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    import sklearn.preprocessing as ps

    #Normalization
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cutouts)
    print('Total spikes: ', estratti_norm.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    print('transformed')
    
    if len(spike_list) != transformed.shape[0]:
        dif = len(spike_list)-transformed.shape[0]
    
    #List of the silhouette scores
    list_score = []   
    
#    n=5
    if clustering=='gerarchico':
        model = AgglomerativeClustering(n_clusters=n, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
    else:
        if clustering== 'gaussianmixture':
            model = GaussianMixture(n_components=n, max_iter=200, random_state=10, tol=0.0001)
        else:
            if clustering=='kmeans':
                model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=400, tol=0.00005, verbose=0, random_state=None, copy_x=True, algorithm='auto')
            else:
                if clustering=='dbscan':
                    model_db = DBSCAN(eps=1.1, min_samples=60, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
                    cluster_labels_db = model_db.fit_predict(transformed)
                    final_list=funzionedscan(cutouts,transformed,cluster_labels_db,spike_list)
    
    if clustering == 'dbscan':
        return final_list
    else:
        cluster_labels = model.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, cluster_labels)
        print("For", n,"clusters, the silhouette score is:", silhouette_avg)
        print('\n')
        #Plot PCA
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        color = []
        for i in cluster_labels:
            color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        if centroids == True:
            print(model.cluster_centers_)
            ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],model.cluster_centers_[:,2],s=300,  c='black', depthshade=False)

        #Plot average waveforms
        ax = fig.add_subplot(1, 2, 2)
        for i in range(n):
            idx = cluster_labels == i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cutouts[idx,:],axis = 0)
            std_wave = np.std(cutouts[idx,:],axis = 0)
            ax.errorbar(range(cutouts[idx,:].shape[1]),mean_wave,yerr = std_wave)
            
        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        plt.show()  
            
        cluster_labels = model.fit_predict(transformed)
        print('Trans shape: ',transformed.shape)
        print('Spike list: ',len(spike_list))
        list_idx = np.unique(cluster_labels)
    
    final_list = []
    
    if len(spike_list)!=transformed.shape[0]:
        spike_list = spike_list[:-dif]
    spike_list=np.array(spike_list)
    for i in list_idx:
        final_list.append(spike_list[cluster_labels==i])        
       
    return final_list
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


def clus(cut,analysis,clustering,spike_list,n,len_data):
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.metrics import silhouette_score
    from scipy.stats import kurtosis
    import numpy as np
    if analysis=='PCA':        
        scale = StandardScaler()
        estratti_norm = scale.fit_transform(cut)
        print('Total spikes: ', estratti_norm.shape[0])
        n_comp=10
        pca = PCA(n_components=n_comp)
        transformed = pca.fit_transform(estratti_norm)
        print('transformed')
        #transformed=cut
    else:
        if analysis=='ICA':
            ica = FastICA(n_components=40)
            ica_components = ica.fit_transform(cut)
            kurtosis_values = kurtosis(ica_components)
            threshold_kurtosis = 3
            selected_components = np.where(kurtosis_values > threshold_kurtosis)[0]
            transformed = [ica_components[:, i] for i in selected_components]
            for i in selected_components:
                print(f'Selected Independent Component {i + 1}: Kurtosis = {kurtosis_values[i]}')

    if clustering=='kmeans':
        num_clusters = n
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(transformed)
        labels = kmeans.labels_
    else:
        if clustering=='dbscan':
            dbscan = DBSCAN(eps=1.5, min_samples=60)
            labels = dbscan.fit_predict(transformed)


    final_data=[]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) == 1:
        print("DBSCAN assigned only one cluster.")
    else:
        silhouette_avg = silhouette_score(transformed, labels)
        num_clusters = len(np.unique(labels[labels != -1]))
        print("For", num_clusters,"clusters, the silhouette score is:", silhouette_avg)

    fig = plt.figure(figsize=(4, 6))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        #final_data.append(spike_list[labels == cluster_label].tolist())

        # Plot the individual cluster data
        plt.subplot(len(unique_labels), 1, i + 1)
        plt.plot(cluster_data.transpose(), alpha=0.5)  # Use alpha for transparency
        #print(cluster_data)
        plt.title(f'Cluster {cluster_label}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')

        # Plot the average waveform
        mean_wave = np.mean(cluster_data, axis=0)
        std_wave = np.std(cluster_data, axis=0)
        plt.plot(mean_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend()

    # Adjust layout to prevent overlapping
    #plt.tight_layout()
    plt.subplots_adjust(hspace=1.5)
    plt.show()
    spike_list=np.array(spike_list)
    for i in unique_labels:
        ul=spike_list[labels==i]
        final_data.append(ul)
        plt.subplot(len(unique_labels), 1, i + 1)
        plt.hist(np.diff(ul), bins=300, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i} numerosity: {len(final_data[i])}, firing rate: {len(final_data[i])*10000/len_data}')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return final_data