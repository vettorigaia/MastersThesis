#LRRK2_MEALibrary
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import sklearn.preprocessing as ps
from sklearn.preprocessing import StandardScaler
from random import randint
from fastdtw import fastdtw
import pymc as pm
from gettext import find
from  McsPy.McsData import RawData
import matplotlib.pyplot as plt
import h5py
import scipy


from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import scipy.stats as st
PPmodelfolder="/Users/Gaia_1/Desktop/tesi/PPplots/"

import glob
#from tqdm import tqdm

def assign_name(file, n_healthy_bl, n_healthy_st, n_lrrk2_bl, n_lrrk2_st):
    target = 1
    stim = 0
    if 'health' in file:
        target = 0
    if 'after' in file:
        stim = 1
    
    if target == 0 and stim == 0:
        name = f"healthy_bl_{n_healthy_bl}"
        n_healthy_bl += 1
    elif target == 0 and stim == 1:
        name = f"healthy_st_{n_healthy_st}"
        n_healthy_st += 1
    elif target == 1 and stim == 0:
        name = f"lrrk2_bl_{n_lrrk2_bl}"
        n_lrrk2_bl += 1
    elif target == 1 and stim == 1:
        name = f"lrrk2_st_{n_lrrk2_st}"
        n_lrrk2_st += 1

    return name, target, stim, n_healthy_bl, n_healthy_st, n_lrrk2_bl, n_lrrk2_st

def spike_sorting(input_path,output_path,savename):
    name_data = input_path.split("/")[-1]
    #file reading:
    print('File Reading...')
    data = h5py.File(input_path,'r')
    data_readings = data['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][()]
    info = data['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][()]
    info_table = pd.DataFrame(info, columns = list(info.dtype.fields.keys()))
    labels = info_table['Label']
    readings = pd.DataFrame(data = data_readings.transpose(), columns = labels)
    fs = 10000 #Sampling Frequency
    print('data shape: ',readings.shape)
    signal_mea=readings.drop([b'Ref'],axis=1)
    ref=readings[b'Ref']
    
    #ref filtering
    freqs, spectrogram = scipy.signal.welch(readings[b'Ref'].values, fs=10000, nfft=1024)
    noise_freq = freqs[spectrogram.argmax()]
    Q = 30
    b, a = scipy.signal.iirnotch(noise_freq, Q, fs)
    Q = 60
    b_2, a_2 = scipy.signal.iirnotch(2*noise_freq, Q, fs)
    channel = readings[b'Ref'].values
    pre_filtered_ref = scipy.signal.filtfilt(b, a, channel)
    pre_filtered_ref = scipy.signal.filtfilt(b_2, a_2, pre_filtered_ref) 
    ref=pre_filtered_ref

    #filtering:
    signal_rows = range(signal_mea.shape[0])
    filt_signal = pd.DataFrame(data = 0, columns=signal_mea.columns, index=signal_rows, dtype = "float32")
    lowcut = 300
    highcut = 3000
    fs=10000
    order=8
    b,a=butter_bandpass(lowcut,highcut,fs,order=order)
    filt_ref=filtfilt(b,a,ref)
    print('\nData Filtering:')
    for x in tqdm(range(signal_mea.shape[1])):
        filt_signal.values[:,x] = scipy.signal.filtfilt(b, a, signal_mea.values[:,x])
    for electrode in signal_mea.columns:
        filt_signal[electrode] = filt_signal[electrode] - filt_ref
    signal_mea=filt_signal
    #detection:
    all_ind=[]
    print('\nSpike Detection: ')
    for i,electrode in enumerate(tqdm(signal_mea.columns)):
        channel=signal_mea[electrode]
        ind=spike_detection(channel)
        all_ind.append(ind)
    #spike extraction:
    cut_outs=[]
    all_new=[]
    print('\nSpike extraction: ')
    for i,electrode in enumerate(tqdm(signal_mea.columns)):
        ind=all_ind[i]
        channel=signal_mea[electrode]
        cut_outs1,all_new1=spike_extraction(ind,channel)
        cut_outs.append(cut_outs1)
        all_new.append(all_new1)    
    # Clustering:
    final_data=[]
    final_firing=[]
    final_firing.append(name_data)
    print('\nClustering: ')
    for channel in (tqdm(range(len(cut_outs)))):
        channel_clusters1,final_firing1=clus(cut_outs[channel],all_new[channel],signal_mea.iloc[:,channel],savename)
        final_data.append(channel_clusters1)
        final_firing.append(final_firing1)
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
        if neuron.shape[0]<max_len:
            diff = max_len-neuron.shape[0]
            adj_neur.append(np.concatenate((neuron,np.zeros([diff]))))
    save_data = 'After'+name_data+'.txt'
    np.savetxt(f"{output_path}/{save_data}.txt", adj_neur, delimiter=', ', fmt='%12.8f')

    print('saved: ',save_data)
    return neurons, final_firing

def spike_detection(data):
    spike_length=30 #3ms (0.003s)
    window_length=600000 #60 sec (1min)
    neg_data=-(data)
    i=0
    ind=[]
    while i < len(data)-window_length:
        neg_window=neg_data[i:i+window_length]
        window=data[i:i+window_length]
        coeff=3
        thresh=coeff*(scipy.stats.median_abs_deviation(window,scale='normal'))
        ind1, _ =find_peaks(neg_window, height=thresh,distance=spike_length)
        last=i
        if len(ind1):
            last=i+ind1[-1]
        ind.extend([index + i for index in ind1])
        i=last+spike_length #0.003 s (30ms)
    window = data[last+spike_length:]
    neg_window=-window
    thresh=coeff*(scipy.stats.median_abs_deviation(window,scale='normal'))
    ind1, _ =find_peaks(neg_window, height=thresh,distance=spike_length)
    ind.extend([index + i for index in ind1])

    firing_rate=len(ind)*10000/len(data)
    print(len(ind), ' spikes detected;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return ind

def spike_extraction(alls,data):
        pre = 0.001
        post = 0.002
        fs=10000
        prima = int(pre*fs)
        dopo = int(post*fs)
        lunghezza_indici = len(alls)
        cut= np.empty([lunghezza_indici, prima+dopo])
        dim = data.shape[0]
        k=0
        signal_std=np.std(data)
        signal_mean=np.mean(data)
        standard_mean=signal_mean
        standard_threshold=signal_std
        for i in alls:
            if (i-prima >= 0) and (i+dopo <= dim):
                spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
                cut[k,:] = spike
                k+=1
        standards=np.std(cut,axis=1)
        means=np.mean(cut,axis=1)
    
        thr1=2*standard_threshold
        thr2=3*standard_mean
    
        indices=np.where((standards<thr1)&(means<thr2))[0]
    
        filtered_alls = np.array(alls)[indices]
        filtered_cut=cut[indices]
        
        spike_means=np.mean(filtered_cut,axis=1,keepdims=True)
        spike_stds=np.std(filtered_cut,axis=1,keepdims=True)
        spike_stds[spike_stds == 0] = 1
        
        standardized_cuts=(filtered_cut-spike_means)/spike_stds
        
        firing_rate=len(indices)*10000/len(data)
        print(len(alls)-len(indices),' spikes removed;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
        return standardized_cuts,filtered_alls 

def clus(cut,spike_list,data,name):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import numpy as np
    save_folder = "/Users/Gaia_1/Desktop/tesi/clustering/"
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
    final_firing=[]
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
        labels=np.zeros(len(spike_list),dtype=int)
    unique_labels=np.unique(labels)
    firings=np.zeros(len(unique_labels))
    color=[]
    for i in labels:
        color.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
    for i,cluster_label in enumerate(unique_labels):
        cluster_data=cut[labels==cluster_label]
        mean_wave=np.mean(cluster_data, axis=0)
        std_wave=np.std(cluster_data, axis=0)
        
        filtered_cluster_data=cluster_data
        
        #plotting_data=filtered_cluster_data.transpose()
        
        firings[i]=len(filtered_cluster_data)*10000/len(data)
        
        fig = plt.figure(figsize=(8,10))
        
        plt.subplot(3,1,i+1)
        #plt.plot(plotting_data,alpha=0.5)
        
        plt.title(f'Cluster {i} \n numerosity: {len(filtered_cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')
        mean_wave = np.mean(filtered_cluster_data, axis=0)
        std_wave = np.std(filtered_cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='blue', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
        plt.show()
        ul=spike_list[labels==i]
        ull=ul
        
        final_data.append(ull)
        final_firing.append(firings)
        plt.subplot(3, 1, i + 1)
        plt.hist(np.diff(ull), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i}, \n firing rate: {format(len(final_data[i])*10000/len(data), ".2f")} Hz')
        plt.show()
        
    return final_data, final_firing

################################POINT PROCESS
def CDF(ISI_data,cdf_2gauss):
    d = np.linspace(min(ISI_data), max(ISI_data), len(ISI_data))
    cdf_model=cdf_2gauss(d)
    bins= np.linspace(min(ISI_data), max(ISI_data), len(ISI_data)+1)
    counts, bins = np.histogram(ISI_data, bins, density=True)
    cdf_emp = np.cumsum(counts * np.diff(bins))
    return cdf_emp,cdf_model

def poiproc(input_path,output_path):
    file_name=input_path.split("/")[-1]
    target=1
    stim=0
    if 'health' in file_name:
        target=0
    if 'KA' in file_name or '24' in file_name:
        stim=1
    
    dataframe = pd.DataFrame()
    counter=0
    list_neurons = np.genfromtxt(file, delimiter=',')
    print('Original number of neurons: ',len(list_neurons))
    for neuron in tqdm(list_neurons):
        neuron=neuron[neuron>0*10000]
        neuron=neuron[neuron<200*10000]
        
        if neuron.shape[0]>1000:
            
            print('  Neuron with ',neuron.shape[0],'spikes')
        else:
            print('    Excluded neuron with n spikes = ',neuron.shape[0])
            continue
        
        ISI_data = np.diff(neuron)/10000
        try:
            map_estimate = Bayesian_mixture_model(ISI_data)
        except Exception as e:
            print(f"An error occured: {e}")
            continue
        if map_estimate!=0:
            counter+=1
            map_estimate['Target']=target
            map_estimate['Stimulation']=stim
            df = pd.DataFrame.from_dict(map_estimate,orient='index')
            dataframe = pd.concat([dataframe,df],axis = 1)
    print('Final number of neurons: ',counter)
    print('Target = ',target)
    final = dataframe.T
    final.to_csv(f'{output_path}/'+file_name)
    return dataframe

def Bayesian_mixture_model(ISI_data):
    save_folder=PPmodelfolder
    with pm.Model() as model:
        ##### WALD DISTRIBUTION (INVERSE GAUSSIAN)
        mu1 = pm.Uniform('mu1',lower=0,upper=0.2)
        lam1 = pm.Uniform('lam1',lower=0.001,upper=0.1)
        obs1 = pm.Wald.dist(mu=mu1,lam=lam1)

        mu2 = pm.Uniform('mu2',lower=0,upper=0.2)
        sigma2 = pm.Uniform('sigma2',lower=0.01,upper=0.7)
        obs2 = pm.TruncatedNormal.dist(mu=mu2, sigma=sigma2, lower=0.0)

        mu3 = pm.Uniform('mu3',lower=0.2,upper=0.6)
        sigma3 = pm.Uniform('sigma3',lower=0.01,upper=0.7)
        obs3 = pm.TruncatedNormal.dist(mu=mu3, sigma=sigma3, lower=0.0)
        
        w = pm.Dirichlet('w', a=np.array([1., 1., 1.]))
        like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2, obs3], observed=ISI_data)
        
    map_estimate = pm.find_MAP(model=model)
    d= np.linspace(0.00, 1, len(ISI_data))
    #d = np.linspace(min(ISI_data), max(ISI_data), len(ISI_data))
    map_estimate['w1'] = map_estimate['w'][0]
    map_estimate['w2'] = map_estimate['w'][1]
    map_estimate['w3'] = map_estimate['w'][2]
    
    mu1=map_estimate['mu1']
    mu2=map_estimate['mu2']
    mu3=map_estimate['mu3']
    lam1=map_estimate['lam1']
    sigma2=map_estimate['sigma2']
    sigma3=map_estimate['sigma3']
    w1=map_estimate['w1']
    w2=map_estimate['w2']
    w3=map_estimate['w3']       
    
    pdf = w1*st.invgauss.pdf(d, mu1/lam1, scale = lam1) + w2*st.norm.pdf(d, mu2, sigma2) + w3*st.norm.pdf(d, mu3, sigma3)
    bins = np.arange(0, .5, 1e-3) 
    
    plt.hist(ISI_data, bins, color='orange', alpha=0.7, label='Histogram',density=True)
    plt.plot(d, pdf, color='blue', label='PDF')
    plt.show()

    cdf_2gauss = lambda x: w1 * st.invgauss.cdf(x,mu1/lam1, scale = lam1) + w2 * st.norm.cdf(x, mu2, sigma2) + w3*st.norm.cdf(x, mu3, sigma3)
    ks_score=st.ks_1samp(ISI_data[ISI_data>0],  cdf_2gauss, method = 'asymp')
    print('ks score: ',ks_score)
    
    # KS plot
    plt.figure(figsize=(12, 8))
    plt.plot(np.sort(ISI_data[ISI_data > 0]), np.linspace(0, 1, len(ISI_data[ISI_data > 0]), endpoint=False), color='orange', label='Empirical CDF')
    plt.plot(d, cdf_2gauss(d), color='blue', label='Model CDF')
    plt.xlabel('Inter-Spike Interval')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF Plot, ksdist={ks_score}, num:{len(ISI_data)}')
    plt.legend()
    plt.show()

    cdf_emp,cdf_model=CDF(ISI_data,cdf_2gauss)
    plt.figure(figsize=(12, 8))
    Nlow = len(ISI_data)  
    # Plot the confidence bounds
    plt.plot([0, 1], [0, 1], 'k:')
    plt.plot([0, 1], [x + 1.36 / np.sqrt(Nlow) for x in [0, 1]], 'r:')
    plt.plot([0, 1], [x - 1.36 / np.sqrt(Nlow) for x in [0, 1]], 'r:')
    plt.plot(cdf_emp, cdf_model)    
    plt.axis([0, 1, 0, 1])         
    plt.xlabel('Model CDF')
    plt.ylabel('Empirical CDF')
    plt.title(f'KS Plot, ksdist={ks_score}, num:{len(ISI_data)}')
    plt.show()

    del map_estimate['w_simplex__']
    del map_estimate['mu1_interval__']
    del map_estimate['lam1_interval__']
    del map_estimate['mu2_interval__']
    del map_estimate['sigma2_interval__']
    del map_estimate['mu3_interval__']
    del map_estimate['sigma3_interval__']
    
    del map_estimate['w']

    print(map_estimate)
    return map_estimate

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
