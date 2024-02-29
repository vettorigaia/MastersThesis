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
from scipy.signal import butter, filtfilt
from scipy import signal



def this_spike_sorting(input_path,output_path,savename):
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
    signal=readings.drop([b'Ref'],axis=1)
    data

    ref=readings[b'Ref']
    #ref=ref[0:750500]
    freqs, spectrogram = signal.welch(readings[b'Ref'].values, fs=10000, nfft=1024)
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
    signal_rows = range(signal.shape[0])
    filt_signal = pd.DataFrame(data = 0, columns=signal.columns, index=signal_rows, dtype = "float32")
    lowcut = 300
    highcut = 3000
    fs=10000
    order=8
    b,a=butter_bandpass(lowcut,highcut,fs,order=order)
    filt_ref=filtfilt(b,a,ref)
    print('Data Filtering:')
    for x in tqdm(range(signal.shape[1])):
        filt_signal.values[:,x] = scipy.signal.filtfilt(b, a, signal.values[:,x])
    for electrode in signal.columns:
        filt_signal[electrode] = filt_signal[electrode] - filt_ref
    signal=filt_signal
    #detection:
    all_ind=[]
    print('Spike Detection: ')
    for i,electrode in enumerate(tqdm(signal.columns)):
        channel=signal[electrode]
        #ind=windowed_spike_detection(channel)
        ind=this_spike_detection(channel)
        all_ind.append(ind)
    #spike extraction:
    cut_outs=[]
    all_new=[]
    print('Spike extraction: ')
    for i,electrode in enumerate(tqdm(signal.columns)):
        ind=all_ind[i]
        channel=signal[electrode]
        cut_outs1,all_new1=spike_extraction(ind,channel)
        cut_outs.append(cut_outs1)
        all_new.append(all_new1)    
    # Clustering:
    final_data=[]
    final_firing=[]
    final_firing.append(name_data)
    print('Clustering: ')
    for channel in (tqdm(range(len(cut_outs)))):
        channel_clusters1,final_firing1=this_clus(cut_outs[channel],all_new[channel],signal.iloc[:,channel],savename)
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
    #np.savetxt("/Users/Gaia_1/Desktop/tesi/Data after SS/%s.txt" % save_data,adj_neur, delimiter=', ', fmt='%12.8f')
    np.savetxt(f"{output_path}/{save_data}.txt", adj_neur, delimiter=', ', fmt='%12.8f')

    print('saved: ',save_data)
    return neurons, final_firing

def poiproc(file,target,stim,):
    dataframe = pd.DataFrame()
    counter=0
    list_neurons = np.genfromtxt(file, delimiter=',')
    counter=0
    print('Original number of neurons: ',len(list_neurons))
    for neuron in tqdm(list_neurons):
        neuron=neuron[neuron>0*10000]
        neuron=neuron[neuron<200*10000]
        #print('  Neuron with ',neuron.shape[0],'spikes')
        if neuron.shape[0]>1000:
            
            print('  Neuron with ',neuron.shape[0],'spikes')
        else:
            print('    Excluded neuron with n spikes = ',neuron.shape[0])
            continue
        
        ISI_healthy = np.diff(neuron)/10000
        map_estimate = Bayesian_mixture_model(ISI_healthy)
        if map_estimate!=0:
            counter+=1
            map_estimate['Target']=target
            map_estimate['Stimulation']=stim
            df = pd.DataFrame.from_dict(map_estimate,orient='index')
            dataframe = pd.concat([dataframe,df],axis = 1)
    print('Final number of neurons: ',counter)
    print('Target = ',target)
    file_name = file.split("/")[-1]
    final = dataframe.T
    final.to_csv('Data after PP/'+file_name)
    return dataframe

def cut_all(alls,data):
    pre = 0.001
    post = 0.002
    fs=10000
    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(alls)
    cut= np.empty([lunghezza_indici, prima+dopo])
    dim = data.shape[0]
    k=0
    #coeff=1.5
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
    
    standardized_cuts=(filtered_cut-spike_means)/spike_stds
    
    firing_rate=len(indices)*10000/len(data)
    print(len(alls)-len(indices),' spikes removed;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return standardized_cuts,filtered_alls
 
def spike_detection(data):
    spike_length=30 #3ms (0.003s)
    window_length=600000 #1 min (60s)
    neg_data=-(data)
    i=0
    ind=[]
    while i < len(data)-window_length:
        neg_window=neg_data[i:i+window_length]
        window=data[i:i+window_length]
        coeff=3
        thresh=coeff*(scipy.stats.median_abs_deviation(window,scale='normal'))
        ind1, peaks =find_peaks(neg_window, height=thresh,distance=spike_length)
        del peaks
        last=i
        if len(ind1):
            last=i+ind1[-1]
        ind.extend([index + i for index in ind1])
        i=last+spike_length #0.003 s (30ms)
    window = data[last+spike_length:]
    neg_window=-window
    thresh=coeff*(scipy.stats.median_abs_deviation(window,scale='normal'))
    ind1, peaks =find_peaks(neg_window, height=thresh,distance=spike_length)
    ind.extend([index + i for index in ind1])# if index + i < len(data)])

    firing_rate=len(ind)*10000/len(data)
    print(len(ind), ' spikes detected;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return ind

################################POINT PROCESS
def Bayesian_mixture_model(ISI_data):
    import scipy.stats as st
    with pm.Model() as model:
        ##### WALD DISTRIBUTION (INVERSE GAUSSIAN)
        mu1 = pm.Uniform('mu1',lower=0.001,upper=0.02)
        lam1 = pm.Uniform('lam1',lower=0.001,upper=0.02)
        obs1 = pm.Wald.dist(mu=mu1,lam=lam1)


        mu2 = pm.Uniform('mu2',lower=0.02,upper=0.04)
        sigma2 = pm.Uniform('sigma2',lower=0.01,upper=0.7)
        obs2 = pm.TruncatedNormal.dist(mu=mu2, sigma=sigma2, lower=0.0)

        mu3 = pm.Uniform('mu3',lower=0.04,upper=0.6)
        sigma3 = pm.Uniform('sigma3',lower=0.01,upper=0.7)
        obs3 = pm.TruncatedNormal.dist(mu=mu3, sigma=sigma3, lower=0.0)
        
        w = pm.Dirichlet('w', a=np.array([1., 1., 1.]))
        like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2, obs3], observed=ISI_data)
        
    map_estimate = pm.find_MAP(model=model)
    d= np.linspace(0.00, 1, len(ISI_data))
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
    #plt.xlim(0, 0.2)
    plt.show()

    cdf_2gauss = lambda x: w1 * st.invgauss.cdf(x,mu1/lam1, scale = lam1) + w2 * st.norm.cdf(x, mu2, sigma2) + w3*st.norm.cdf(x, mu3, sigma3)
    ks_score=st.ks_1samp(ISI_data[ISI_data>0],  cdf_2gauss, method = 'asymp')
    print('ks score: ',ks_score)
    
    del map_estimate['w_simplex__']
    del map_estimate['mu1_interval__']
    del map_estimate['lam1_interval__']
    del map_estimate['mu2_interval__']
    del map_estimate['sigma2_interval__']
    del map_estimate['mu3_interval__']
    del map_estimate['sigma3_interval__']
    
    del map_estimate['w']

    print(map_estimate)
    if ks_score[0]>0.3:
        map_estimate=0
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
        #distances=np.abs(cluster_data - mean_wave)
        #distance_threshold=2*std_wave
        #indices_to_keep=np.all(distances<=distance_threshold,axis=1)
        #filtered_cluster_data=cluster_data[indices_to_keep]
        filtered_cluster_data=cluster_data
        plotting_data=filtered_cluster_data.transpose()
        firings[i]=len(filtered_cluster_data)*10000/len(data)
        fig = plt.figure(figsize=(8,10))
        plt.subplot(3,1,i+1)
        plt.plot(plotting_data,alpha=0.5)
        plt.title(f'Cluster {i} \n numerosity: {len(filtered_cluster_data)}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal Amplitude')
        mean_wave = np.mean(filtered_cluster_data, axis=0)
        std_wave = np.std(filtered_cluster_data, axis=0)
        plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, color='black', linewidth=2, label='Avg. Waveform')
        plt.legend(loc='lower right')
        plt.show()
        ul=spike_list[labels==i]
        ull=ul
        #ull=ul[indices_to_keep]
        final_data.append(ull)
        final_firing.append(firings)
        plt.subplot(3, 1, i + 1)
        plt.hist(np.diff(ull), bins=100, density=True, alpha=0.5, color='blue', edgecolor='black')
        plt.title(f'ISI: Cluster {i}, \n firing rate: {format(len(final_data[i])*10000/len(data), ".2f")} Hz')
        plt.show()
        
    return final_data, final_firing


'''
def new_cut(all,data):
    pre = 0.0015
    post = 0.0015
    fs=10000
    prima = int(pre*fs)
    dopo = int(post*fs)
    lunghezza_indici = len(all)
    cut= np.empty([lunghezza_indici, prima+dopo])
    dim = data.shape[0]
    k=0
    coeff=1.5
    signal_std=np.std(data)
    standard_threshold=signal_std
    for i in all:
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            cut[k,:] = spike
            k+=1
    standards=np.std(cut)
    indices_to_keep=np.all(standards<coeff*standard_threshold,axis=1)
    all_new=indices_to_keep
    filtered_cut=cut[indices_to_keep]
    k=0
    cut=np.empty([indices_to_keep, prima+dopo])
    for spike in filtered_cut:
        media=(np.mean(spike))
        std=np.std(spike)
        spike_std=(spike-media)/std
        cut[k,:]=spike_std
        k+=1
    firing_rate=len(all_new)*10000/len(data)
    print(len(all)-len(all_new),' spikes removed;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return cut,all_new

def find_all_spikes(data,thresh):
    spike_length=30 #3ms
    ind, peaks_amp = scipy.signal.find_peaks(abs(data), height=thresh, distance= spike_length)
    pos=peaks_amp['peak heights'][peaks_amp['peak heights']>0]
    neg=peaks_amp['peak heights'][peaks_amp['peak heights']<0]
    firing_rate=(len(ind)*10000)/len(data)
    #print(len(ind), ' spikes detected;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return ind

def windowed_spike_detection(data):
    spike_length=30 #3ms
    window_length=10000 #1 sec
    abs_data=abs(data)
    i=0
    ind=[]
    while i < len(data)+window_length:
        abs_window=abs_data[i:i+window_length]
        window=data[i:i+window_length]
        #if abso==0:
            #thresh=coeff*(scipy.stats.median_abs_deviation(window,scale='normal'))
        #else:
        coeff=4
        thresh=coeff*(scipy.stats.median_abs_deviation(abs_window,scale='normal'))
        ind1, peaks =find_peaks(abs_window, height=thresh,distance=spike_length)
        last=i
        if len(ind1):
            last=i+ind1[-1]
        ind.extend([index + i for index in ind1])
        i=last+spike_length
    firing_rate=len(ind)*10000/len(data)
    print(len(ind), ' spikes detected;  ', 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return ind

def neg_RMM(data):
    # window size 100ms, threshold for first spike: 4*mad(window), threshold for second spike: 1.1 mean(window)
    # differential threshold: 7*mad(window)
    window_size=1000 #0.1 sec (100ms)
    spike_length=300 #0.03sec (30ms)
    i=0
    first_peaks_set=set()
    second_peaks_set=set()
    abs_data=abs(data)
    #pbar = tqdm(total = len(data)-window_size)
    neg_data=-(data)

   # with tqdm(total=100) as pbar:
    while i<=len(data)-window_size-50:

        i_bf=i
        found=False
        abs_window=abs_data[i:i+window_size]
        window=data[i:i+window_size]
        neg_window=neg_data[i:i+window_size]
        mad=scipy.stats.median_abs_deviation(window,scale='normal')
        thresh=4*mad
        media=1.1*np.mean(window)
        first_peaks,amp=find_peaks(neg_window,height=float(thresh),distance=spike_length)
        first_amps=amp['peak_heights']
        len1=len(first_peaks)
        second_peaks,amp=find_peaks(window.ravel(),height=float(media))
        second_amps=amp['peak_heights']
        len2=len(second_peaks)
        if len1==0 or len2==0:
            i=i+spike_length
        else:
            for k in range(len(first_peaks)):
                for j in range(len(second_peaks)):
                    if second_peaks[j]>first_peaks[k]:
                        #cioè se il secondo picco è successivo al primo
                        primo=first_amps[k]
                        secondo=second_amps[j]
                        #diff=abs_window[first_peaks[0][k]] + abs_window[second_peaks[0][j]]
                        diff=primo+secondo

                        if found==False and diff > 7*mad:
                            #cioè se non sono stati trovati già due indici che soddisfano e hanno una distanza sopra la soglia
                            peaks_indices=(i+first_peaks[k],i+second_peaks[j])
                            found=True
                            #print('found',i)
                            first_peaks_set.add(peaks_indices[0])
                            second_peaks_set.add(peaks_indices[1])
                            i=peaks_indices[1]+1
            first_peaks=[]
            second_peaks=[]
            i=i+spike_length
        #time.sleep(0.1)  # Adjust the sleep duration as needed
        delta=i-i_bf
        #pbar.update(delta)
    minima=sorted(list(first_peaks_set))
    maxima=sorted(list(second_peaks_set))
    firing_rate=len(minima)*10000/len(data)
    print('detected spikes:', len(minima), len(maxima), 'firing rate: {:.2f}'.format(firing_rate),'Hz')
    return minima

def RMM(data):
    # window size 30ms, threshold for first spike: 3*mad(window), threshold for second spike: 1.1 mean(window)
    # differential threshold: 6*mad(window)
    window_size=10000 #1 sec (1000ms)
    research_size=300 #0.03 sec (30ms)
    i=0
    first_peaks_set=set()
    second_peaks_set=set()
    neg_data=-(data)
    pbar = tqdm(total = len(data)-window_size)
    overlap=0
   # with tqdm(total=100) as pbar:
    while i<=len(data)-window_size-50:

        i_bf=i
        entrato=False
        found=False
        neg_window=neg_data[i:i+window_size-overlap*window_size]
        window=data[i:i+window_size-overlap*window_size]
        mad=scipy.stats.median_abs_deviation(window)
        thresh=4*mad
        media=1.1*np.mean(window)
        first_peaks,amp=find_peaks(neg_window.ravel(),height=float(thresh))
        len1=len(first_peaks[0])
        for k in len(first_peaks):
            starting_point=i+k
            research_window=data[starting_point:starting_point+research_size]
            second_peaks,amp2=find_peaks(research_window,height=float(media))
            if first_peaks[k+1]>second_peaks[0]:
                second=second_peaks[0]
                first_peaks_set.append(first_peaks[k])
            overlap=1

        while j <=len(research_window)
        second_peaks=find_peaks(window.ravel(),height=float(media))
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
def select_clus(cut,spike_list,data):
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
        distances=np.abs(cluster_data - mean_wave)
        distance_threshold=2*std_wave
        indices_to_keep=np.all(distances<=distance_threshold,axis=1)
        filtered_cluster_data=cluster_data[indices_to_keep]
        plotting_data=filtered_cluster_data.transpose()
        firings[i]=len(filtered_cluster_data)*10000/len(data)
        plt.subplot(3,1,i+1)
        plt.plot(plotting_data,alpha=0.5)
        plt.title(f'Cluster {i} \n numerosity: {len(filtered_cluster_data)}')
        plt.xlabel('Time [ms]')
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
    if len(unique_labels)>10:
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=color, alpha=0.8, s=10, marker='.')
        ax = fig.add_subplot(1, 2, 2)
        for i,cluster_label in enumerate(unique_labels):    
            idx=cluster_label==i
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            mean_wave = np.mean(cut[idx,:],axis = 0)
            std_wave = np.std(cut[idx,:],axis = 0)
            #ax.errorbar(range(cut[idx,:].shape[1]),mean_wave,yerr = std_wave)
            plt.errorbar(range(mean_wave.shape[0]), mean_wave, yerr=std_wave, c=color)
            #ax.errorbar(range(mean_wave.shape[0]),mean_wave,yerr = std_wave)

        plt.xlabel('Time [0.1ms]')
        plt.ylabel('Voltage [\u03BCV]')
        plt.show()    

    return final_data

'''