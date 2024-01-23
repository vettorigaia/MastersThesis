#SORTING Ver2
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
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
import scipy
import pywt
#from tqdm import tqdm
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import time



def spike_sorting(name_data,complete_string,threshold,clustering,coeff,c1):
    #file reading:
    data = h5py.File(complete_string,'r')
    data_readings = data['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][()]
    info = data['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][()]
    info_table = pd.DataFrame(info, columns = list(info.dtype.fields.keys()))
    labels = info_table['Label']
    readings = pd.DataFrame(data = data_readings.transpose(), columns = labels)
    fs = 10000 #Sampling Frequency
    print('data shape: ',readings.shape)
    prova=readings.drop([b'Ref'],axis=1)
    #prova=prova.iloc[inizio:fine, :10]
    prova=prova.iloc[:, :10]
    ref=readings[b'Ref']
    #ref=ref[inizio:fine]
    #filtering:
    prova_rows = range(prova.shape[0])
    filt_prova = pd.DataFrame(data = 0, columns=prova.columns, index=prova_rows, dtype = "float32")
    lowcut = 300
    highcut = 3000
    fs=10000
    order=8
    b,a=butter_bandpass(lowcut,highcut,fs,order=order)
    filt_ref=filtfilt(b,a,ref)
    for x in tqdm(range(prova.shape[1])):
        filt_prova.values[:,x] = scipy.signal.filtfilt(b, a, prova.values[:,x])
    for electrode in prova.columns:
        filt_prova[electrode] = filt_prova[electrode] - filt_ref
    prova=filt_prova
    #detection:
    if threshold==0:
        threshold=[]
        for i,electrode in tqdm(enumerate(prova.columns)):
            threshold.append(coeff*(scipy.stats.median_abs_deviation(prova[electrode].values,scale='normal')))            
    all_ind=[]
    for i,electrode in tqdm(enumerate(prova.columns)):
        channel=prova[electrode]
        thresh=threshold[i]
        ind=find_all_spikes(channel,thresh)
        all_ind.append(ind)
    #spike extraction:
    cut_outs=[]
    all_new=[]
    for i,electrode in enumerate(tqdm(prova.columns)):
        ind=all_ind[i]
        channel=prova[electrode]
        cut_outs1,all_new1=cut_all(ind,channel,c1)
        cut_outs.append(cut_outs1)
        all_new.append(all_new1)    
    # Clustering:
    final_data=[]
    if clustering=='kmeans':
            for channel in (tqdm(range(len(cut_outs)))):
                #channel_clusters1=comparative_clus(cut_outs[channel],all_new[channel],prova.iloc[:,channel])
                channel_clusters1=clus(cut_outs[channel],all_new[channel],prova.iloc[:,channel])
                final_data.append(channel_clusters1)
    elif clustering=='dbscan':
            for channel in (tqdm(range(len(cut_outs)))):
                eps=int(scipy.stats.median_abs_deviation(prova.iloc[:,channel])/2)
                channel_clusters1=dbscan_clustering(cut_outs[channel],all_new[channel],prova.iloc[:,channel],eps)
                final_data.append(channel_clusters1)
    neurons=[]
    for channel in final_data:
        for neuron in channel:
            neurons.append(neuron)
    print(len(neurons),' neurons detected and sorted')
    adj_neur=[]
    counter = 0
    max_len=0
    for neuron in neurons:
        print('counter: ',counter,neuron.shape[0])
        if neuron.shape[0]>max_len:
            max_len=neuron.shape[0]
        counter+=1
    for neuron in neurons:
        if neuron.shape[0]<max_len and neuron.shape[0]>=1000:
            diff = max_len-neuron.shape[0]
            adj_neur.append(np.concatenate((neuron,np.zeros([diff]))))
    save_data = 'After'+name_data+'.txt'
    np.savetxt("/Users/Gaia_1/Desktop/tesi/Data after SS/%s.txt" % save_data,adj_neur, delimiter=', ', fmt='%12.8f')
    print(save_data)
    return neurons

def poiproc(neurons,target,stim):
    from scipy.stats import ks_2samp
    dataframe = pd.DataFrame()
    #counter_net=1
    counter=0
    #for net in list_dir_ok:
    #print(counter_net,') ',net)
    #counter_net+=1
    list_neurons = neurons #np.genfromtxt(net, delimiter=',')
    counter=0
    print('Original number of neurons: ',len(list_neurons))
    for neuron in list_neurons:
        neuron=neuron[neuron>0*10000]
        neuron=neuron[neuron<200*10000]
        print('  Neuron with ',neuron.shape[0],'spikes')
        if neuron.shape[0]>1000:
            
            counter+=1
        else:
            print('    Excluded neuron with n spikes = ',neuron.shape[0])
            continue
        
        ISI_healthy = np.diff(neuron)/10000
        map_estimate = Bayesian_mixture_model(ISI_healthy)
        map_estimate['Target']=target
        map_estimate['Stimulation']=stim
        df = pd.DataFrame.from_dict(map_estimate,orient='index')
        dataframe = pd.concat([dataframe,df],axis = 1)
    print('Final number of neurons: ',counter)
    print('Target = ',target)
    return dataframe



    ks_2samp(lista_samples,ISI_healthy,mode = 'asymp')


def cut_all(all,data,c1):
    pre = 0.0015
    post = 0.0015
    fs=10000
    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(all)
    cut= np.empty([lunghezza_indici, prima+dopo])
    dim = data.shape[0]
    k=0
    signal_std=np.std(data)
    all_new=[]
    for i in all:
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            std=np.std(spike)
            spike_std=(spike-media)/std
            if abs(std)<=c1*abs(signal_std):
                cut[k,:] = spike_std
                all_new.append(i)
                k += 1
    size=k
    cut=cut[0:size]
    firing_rate=len(all_new)*10000/len(data)
    print(len(all)-len(all_new),' spikes removed;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return cut,all_new
def find_all_spikes(data,thresh):
    spike_length=30 #3ms
    ind, peaks_amp = scipy.signal.find_peaks(abs(data), height=thresh, distance= spike_length)
    firing_rate=(len(ind)*10000)/len(data)
    print(len(ind), ' spikes detected;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return ind
def clus(cut,spike_list,data):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import numpy as np
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('\n______________________________________________________________________________________________________________')
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=3
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    spike_list=np.array(spike_list)
    kmeans_score=[]
    final_data=[]
    for n in range (2,4):
        model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=400, tol=0.25, verbose=0, random_state=None, copy_x=True,  algorithm='lloyd')
        labels = model.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, labels)
        kmeans_score.append(silhouette_avg)
    top_clusters_kmeans = kmeans_score.index(max(kmeans_score))+2
    if max(kmeans_score)>=0.4:
        print("\n\n\033[1;31;47mBest cluster in the range 1 to 3: ",top_clusters_kmeans,", with a silhouette score of: ",max(kmeans_score), "\u001b[0m  \n\n")
        model = KMeans(n_clusters=top_clusters_kmeans,  init='k-means++', n_init=10, max_iter=400, tol=0.25, verbose=0, random_state=None, copy_x=True,  algorithm='lloyd')
        labels = model.fit_predict(transformed)
    else:
        print('Clustering algorithm detected only one cluster')
        labels=np.zeros(len(spike_list))
    unique_labels=np.unique(labels)
    firings=np.zeros(len(unique_labels))
    for i,cluster_label in enumerate(unique_labels):
        cluster_data=cut[labels==cluster_label]
        mean_wave=np.mean(cluster_data, axis=0)
        std_wave=np.std(cluster_data, axis=0)
        distances=np.abs(cluster_data - mean_wave)
        distance_threshold=4*std_wave
        indices_to_keep=np.all(distances<=distance_threshold,axis=1)
        filtered_cluster_data=cluster_data[indices_to_keep]
        plotting_data=filtered_cluster_data.transpose()
        firings[i]=len(filtered_cluster_data)*10000/len(data)
        plt.subplot(3,1,i+1)
        plt.plot(plotting_data,alpha=0.5)
        plt.title(f'Cluster {i} \n numerosity: {len(filtered_cluster_data)}')
        plt.xlabel=('Time [ms]')
        plt.ylabel('Signal Amplitude')
        mean_wave = np.mean(filtered_cluster_data, axis=0)
        std_wave = np.std(filtered_cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
        plt.show()
        ul=spike_list[labels==i]
        ull=ul[indices_to_keep]
        final_data.append(ull)
        plt.subplot(3, 1, i + 1)
        plt.hist(np.diff(ull), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i}, \n firing rate: {format(len(final_data[i])*10000/len(data), ".2f")} Hz')
        plt.show()
    return final_data

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
        like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2, obs3], observed=ISI_data)

        #step = pm.NUTS(target_accept=0.9)
        #trace = pm.sample(step=step,draws=1000,chains=1,tune=1000,cores=4)
        
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


    return map_estimate
#######################







def comparative_clus(cut,spike_list,data):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import numpy as np
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('\n______________________________________________________________________________________________________________')
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=3
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    spike_list=np.array(spike_list)
    kmeans_score=[]
    final_data=[]
    for n in range (2,4):
        model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=400, tol=0.00005, verbose=0, random_state=None, copy_x=True,  algorithm='lloyd')
        labels = model.fit_predict(transformed)
        silhouette_avg = silhouette_score(transformed, labels)
        kmeans_score.append(silhouette_avg)
    top_clusters_kmeans = kmeans_score.index(max(kmeans_score))+2
    if max(kmeans_score)>=0.2:
        print("\n\n\033[1;31;47mBest cluster in the range 1 to 3: ",top_clusters_kmeans,", with a silhouette score of: ",max(kmeans_score), "\u001b[0m  \n\n")
        model = KMeans(n_clusters=top_clusters_kmeans, init='k-means++', n_init=10, max_iter=400, tol=0.25,  verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
        labels = model.fit_predict(transformed)
    else:
        print('Clustering algorithm detected only one cluster')
        labels=np.zeros(len(spike_list))
    unique_labels=np.unique(labels)
    firings=np.zeros(len(unique_labels))
    fig = plt.figure(figsize=(6, 8))
    for i, cluster_label in enumerate(unique_labels):
        cluster_data = cut[labels == cluster_label]
        plotting_data=cluster_data.transpose()
        firings[i]=len(cluster_data)*10000/len(data)
        plt.subplot(3,1, i + 1)
        plt.plot(plotting_data, alpha=0.5)  # Use alpha for transparency
        plt.title(f'Cluster {i} \n numerosity: {len(cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')
        mean_wave = np.mean(cluster_data, axis=0)
        std_wave = np.std(cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
        plt.show()
        ul=spike_list[labels==i]
        final_data.append(ul)
        plt.subplot(3, 1, i + 1)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i}, \n firing rate: {format(len(final_data[i])*10000/len(data), ".3f")}')
        plt.show()
    return final_data
def three_spike_sorting(baseline,stimulation,post,clustering):
    neurons_BL,threshold=spike_sorting(baseline,0,clustering)
    neurons_stim,threshold0=spike_sorting(stimulation,threshold,clustering)
    neurons_post,threshold0=spike_sorting(post,threshold,clustering)
    print(len(neurons_BL),' BASELINE neurons detected and sorted')
    print(len(neurons_stim),' during-stimulation neurons detected and sorted')
    print(len(neurons_post),' 24hrs-after neurons detected and sorted')
    return neurons_BL,neurons_stim,neurons_post
def spike_sorting_separate(complete_string,threshold,clustering,coeff,c1):
    #file reading:
    data = h5py.File(complete_string,'r')
    data_readings = data['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][()]
    info = data['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][()]
    info_table = pd.DataFrame(info, columns = list(info.dtype.fields.keys()))
    labels = info_table['Label']
    readings = pd.DataFrame(data = data_readings.transpose(), columns = labels)
    fs = 10000 #Sampling Frequency
    print('data shape: ',readings.shape)
    prova=readings.drop([b'Ref'],axis=1)
    prova=prova.iloc[:, :10]
    ref=readings[b'Ref']
    #filtering:
    prova_rows = range(prova.shape[0])
    filt_prova = pd.DataFrame(data = 0, columns=prova.columns, index=prova_rows, dtype = "float32")
    lowcut = 300
    highcut = 3000
    fs=10000
    order=8
    b,a=butter_bandpass(lowcut,highcut,fs,order=order)
    filt_ref=filtfilt(b,a,ref)
    for x in tqdm(range(prova.shape[1])):
        filt_prova.values[:,x] = scipy.signal.filtfilt(b, a, prova.values[:,x])
    for electrode in prova.columns:
        filt_prova[electrode] = filt_prova[electrode] - filt_ref
    prova=filt_prova
    #detection:
    if threshold==0:
        threshold=[]
        for i,electrode in tqdm(enumerate(prova.columns)):
            threshold.append(coeff*(scipy.stats.median_abs_deviation(prova[electrode].values)))            
    pos_ind=[]
    neg_ind=[]
    for i,electrode in tqdm(enumerate(prova.columns)):
        channel=prova[electrode]
        thresh=threshold[i]
        pos,neg=new_find_all_spikes(channel,thresh)
        pos_ind.append(pos)
        neg_ind.append(neg)

    #spike extraction:
    pos_cut=[]
    neg_cut=[]
    n_pos=[]
    n_neg=[]
    for i,electrode in enumerate(tqdm(prova.columns)):
        pos=pos_ind[i]
        neg=neg_ind[i]
        channel=prova[electrode]
        print(electrode,':')
        pos_cut1,n_pos1, neg_cut1,n_neg1 = cut(pos,neg,channel,c1)
        pos_cut.append(pos_cut1)
        neg_cut.append(neg_cut1)
        n_pos.append(n_pos1)
        n_neg.append(n_neg1)
    
    # Clustering:
    final_data=[]
    for channel in (tqdm(range(len(pos_cut)))):
        if clustering=='kmeans':
            channel_clusters1=comparative_clus(pos_cut[channel],n_pos[channel],prova.iloc[:,channel])
            channel_clusters2=comparative_clus(neg_cut[channel],n_neg[channel],prova.iloc[:,channel])
        elif clustering=='dbscan':
            eps=int(scipy.stats.median_abs_deviation(prova.iloc[:,channel])/2)
            channel_clusters1=dbscan_clustering(pos_cut[channel],n_pos[channel],prova.iloc[:,channel],eps)
            channel_clusters2=dbscan_clustering(neg_cut[channel],n_neg[channel],prova.iloc[:,channel,eps])
        final_data.append(channel_clusters1)
        final_data.append(channel_clusters2)
    neurons=[]
    for neuron in final_data:
        neurons.append(neuron)
    print(len(neurons),' neurons detected and sorted')
    return neurons,threshold
def dbscan_clustering(cut,spike_list,data,eps0):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import numpy as np
    import math

    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    n_comp=3
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)
    #transformed=cut
    dbscan = DBSCAN(min_samples=15, eps=eps0, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, n_jobs=None)
    labels = dbscan.fit_predict(transformed)


    final_data=[]
    temporary_data=[]
    unique_labels = np.unique(labels)
    print('labels: ',unique_labels)

    if len(unique_labels) == 1:
        print("DBSCAN assigned only one cluster.")
        
        cluster_data=cut
        fig = plt.figure(figsize=(4, 5))
        plt.plot(cluster_data.transpose(), alpha=0.5)
        plt.title(f'Cluster 0')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')
        mean_wave = np.mean(cluster_data, axis=0)
        std_wave = np.std(cluster_data, axis=0)
        plt.plot(mean_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
        plt.show()
        spike_list=np.array(spike_list)
        ul=spike_list[labels==-1]
        final_data.append(ul)
        plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {0} \n numerosity: {len(final_data[0])}, \n firing rate: {len(final_data[0])*10000/len(data)}')
        plt.show()
        

    else:
        silhouette_avg = silhouette_score(transformed, labels)
        num_clusters = len(np.unique(labels[labels != -1]))
        print("For", num_clusters,"clusters, the silhouette score is:", silhouette_avg)

        fig = plt.figure(figsize=(6, 8))

        # Iterate over unique cluster labels
        for i, cluster_label in enumerate(unique_labels):
            # Extract data points for the current cluster
            cluster_data = cut[labels == cluster_label]
            plotting_data=cluster_data.transpose()

            # Plot the individual cluster data
            quad = math.ceil(len(unique_labels)/2)
            plt.subplot(3, 1, i + 1)
            plt.plot(plotting_data, alpha=0.5)
            plt.title(f'Cluster {cluster_label} index {i}')
            plt.xlabel('Time [ms]')
            plt.ylabel('Signal Amplitude')

            # Plot the average waveform
            mean_wave = np.mean(cluster_data, axis=0)
            std_wave = np.std(cluster_data, axis=0)
            plt.plot(mean_wave, color='black', linewidth=2, label='Avg. Waveform')
            plt.legend(loc='lower left')

        plt.subplots_adjust(hspace=0.5)
        plt.show()
        spike_list=np.array(spike_list)
        for i in unique_labels:
            ul=spike_list[labels==i]
            temporary_data.append(ul)
            if i !=-1:
                final_data.append(ul)
            plt.subplot(3,1, i + 2)
            plt.hist(np.diff(ul), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
            plt.title(f'ISI: Cluster {i} \n numerosity: {len(temporary_data[i+1])}, \n firing rate: {len(temporary_data[i+1])*10000/len(data)}')
        plt.subplots_adjust(hspace=2)
        plt.show()
    return final_data
def old_clus(cut,clustering,spike_list,data,flag=0):
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
            if silhouette_avg<0.06:
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

    del(unique_labels)
    return final_data
def nested_clus(cut,clustering,spike_list,data,flag=0,count=0):
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
    n_comp=6
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
            if silhouette_avg<0.06:
                break       
            if silhouette_avg>0.3:
                flag=1
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
    
    fig = plt.figure(figsize=(6, 3))

    # Iterate over unique cluster labels
    for i, cluster_label in enumerate(unique_labels):
        # Extract data points for the current cluster
        cluster_data = cut[labels == cluster_label]
        firings[i]=len(cluster_data)*10000/len_data
        
        # Plot the individual cluster data
        if len(unique_labels)<=2:
            size1=1
            size2=len(unique_labels)
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
    firing_threshold=0.08
    print('firing rate threshold: ',firing_threshold)
    info.append('firing threshold')
    info.append(firing_threshold)
    info.append('mean firing')
    info.append(mean_firing)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    fig = plt.figure(figsize=(6, 3))
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
    print('flag: ',flag)
    if top_clusters==2 and flag==0:
        count+=1
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
        
        print('sub-clusters size',len(cut0),len(cut1))
        if count>=2 or len(cut0)<=1000 or len(cut1)<=1000:
            flag=1
        print('first sub-clustering, count: ',count)
        del final_data
        final_data=nested_clus(cut0,clustering,spike_list0,data,flag,count)
        count=0
        print('second sub-clustering, count: ',count)
        final_data1=nested_clus(cut1,clustering,spike_list1,data,flag,count)
        for arr in final_data1:
            final_data.append(arr)

    else:
        del(unique_labels)
    return final_data
def old_find_all_spikes(data,thresh):
    pos=[]
    neg=[]
    neg_data=-data
    window_size=10000 #(0.1 s)
    pbar = tqdm(total = len(data)-window_size)
    i=0
    i_bf=0
    while i <len(data)-window_size-11:
        window=data[i:i+window_size]
        neg_window=neg_data[i:i+window_size]
        mad=scipy.stats.median_abs_deviation(window)
        thresh=3*mad
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
            i=i+max(mp,mn)+10#0.0030*10000=30 s
        delta=i-i_bf
        pbar.update(delta)
    pbar.refresh()

    firing_rate=(len(pos)+len(neg))*10000/len(data)
    print('positive spikes',len(pos),'negative spikes',len(neg),'detected spikes:', len(pos)+len(neg), 'firing rate: ',firing_rate)
    return pos,neg
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
def bounded_clus(n_comp,n_min,n_tries,cut,clustering,spike_list,data):
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
    #n_min=int((10000*len(spike_list))/(2*len_data))
    #n_tries=int(((10000*len(spike_list))/(0.16*len_data)+1))
    print('cluster range: ',n_min,n_tries-1)
    scale = StandardScaler()
    estratti_norm = scale.fit_transform(cut)
    print('Total spikes: ', estratti_norm.shape[0])
    #n_comp=10
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(estratti_norm)

    info=[]
    list_score=[]
    DB_score=[]
    best_score=[]
    #CH_score=[]
    if clustering=='kmeans':
        for n in range (n_min,n_tries):
            model = KMeans(n_clusters=n, n_init='auto', copy_x=True, algorithm='lloyd')
            labels = model.fit_predict(transformed)
            if (n != 1):
                silhouette_avg = silhouette_score(transformed, labels)
                #CH=metrics.calinski_harabasz_score(transformed, labels)
                DB=metrics.davies_bouldin_score(transformed, labels)
                print("For", n,"clusters, the silhouette score is:", format(silhouette_avg, ".3f"),'DB score',format(DB, ".3f"))
                list_score.append(silhouette_avg)
                DB_score.append(DB/10)
                best_score.append(silhouette_avg-(DB/10))
                #CH_score.append(CH/3000)
                #best_score.append(silhouette_avg)
                del(model)
                del(labels)
        top_clusters = (best_score.index(max(best_score)))+n_min
        #top_DB=(DB_score.index(max(DB_score)))+n_min
        #top_clusters=(DB_score.index(min(DB_score)))+n_min
        #print('top cluster: ',(DB_score.index(min(DB_score))),'+',n_min,'=',top_clusters)
        print('optimal cluster number: ',top_clusters)
        
        plt.figure()
        #x=np.arange(n_min,n_tries)
        plt.plot(list_score)
        plt.plot(DB_score)
        plt.plot(best_score)
        plt.title('Clustering scores')
        plt.xlabel('Number of clusters')
        plt.legend(loc='lower right')
        #plt.ylim(0,2)
        plt.scatter((list_score.index(max(list_score))), max(list_score), c='red', marker='o', label='Maximum s score')
        plt.scatter(DB_score.index(min(DB_score)), min(DB_score), c='green', marker='o', label='Minimum DB score')
        #plt.scatter((CH_score.index(max(CH_score))), max(CH_score), c='blue', marker='o', label='Maximum CH score')
        plt.scatter((best_score.index(max(best_score))), max(best_score), c='blue', marker='x', label='Best sil - DB score')
        plt.show()

        #top_clusters=num_clusters
        print("\n\n\033[1;31;47mBest cluster in the range", n_min, "to ", n_tries-1, "is : ",top_clusters,", with a silhouette score of: ",list_score[top_clusters-n_min],"\u001b[0m  \n\n")#,'DB score: ',DB_score[top_clusters-n_min],plt_value, "\u001b[0m  \n\n")

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
                #DB_score.append(DB)
                #best_score.append(2*silhouette_avg-DB)
                best_score.append(silhouette_avg)
                del(u)
                del(labels)

        top_clusters = (best_score.index(max(best_score)))+n_min
        plt.figure()
        #x=np.arange(n_min,n_tries)
        plt.plot(list_score)
        #plt.plot(DB_score)
        #plt.plot(CH_score)
        plt.title('Silhouette score')
        plt.xlabel('Number of clusters')
        #plt.xlabel('x')
        #plt.ylim(0,2)
        plt.scatter((list_score.index(max(list_score))), max(list_score), c='red', marker='o', label='Maximum s score')
        #plt.scatter(plt_index, plt_value, c='green', marker='o', label='Minimum DB score')
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
    firing_threshold=0#mean_firing-std_firing
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
def old_new_find_all_spikes(data,threshold):
    spike_length=30 #3ms
    ind, peaks_amp = scipy.signal.find_peaks(abs(data), height=threshold, distance= spike_length)
    ind_pos = data[ind] < 0 
    ind_neg = data[ind] > 0
    pos = ind[ind_pos]
    neg = ind[ind_neg]
    firing_rate=(len(pos)+len(neg))*10000/len(data)
    print('positive spikes', len(pos), 'negative spikes', len(neg), 'detected spikes:', len(pos) + len(neg), 'firing rate: {:.2f}'.format(firing_rate))
    return pos, neg
def old_cut(pos,neg,data,c1):
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
    signal_std=np.std(data)
    pos_new=[]
    all_new=[]
    for i in pos:
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            std=np.std(spike)
            spike_std=(spike-media)/std
            if abs(std)<=c1*abs(signal_std) :
                pos_cut[k,:] = spike_std
                pos_new.append(i)
                all_new.append(i)
                k += 1
    possize=k
    pos_cut=pos_cut[0:possize]
    k=0
    neg_new=[]
    for i in neg:
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            std=np.std(spike)
            spike_std=(spike-media)/std
            if abs(std)<=c1*abs(signal_std) :
                neg_cut[k,:] = spike_std
                neg_new.append(i)
                k  += 1
    negsize=k
    neg_cut=neg_cut[0:negsize]
    firing_rate=(len(pos_new)+len(neg_new))*10000/len(data)
    print('positive spikes removed', len(pos)-len(pos_new), 'negative spikes removed: ', len(neg)-len(neg_new), 'total spikes :', len(pos_new) + len(neg_new), 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return pos_cut,pos_new,neg_cut,neg_new


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