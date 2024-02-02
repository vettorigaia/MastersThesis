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



def spike_sorting(input_path,output_path):
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
    prova=readings.drop([b'Ref'],axis=1)
    #prova=prova.iloc[inizio:fine, :10]
    #prova=prova.iloc[:, :15]
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
    print('Data Filtering:')
    for x in tqdm(range(prova.shape[1])):
        filt_prova.values[:,x] = scipy.signal.filtfilt(b, a, prova.values[:,x])
    for electrode in prova.columns:
        filt_prova[electrode] = filt_prova[electrode] - filt_ref
    prova=filt_prova
    #detection:
    all_ind=[]
    print('Spike Detection: ')
    for i,electrode in enumerate(tqdm(prova.columns)):
        channel=prova[electrode]
        ind=windowed_spike_detection(channel)
        all_ind.append(ind)
    #spike extraction:
    cut_outs=[]
    all_new=[]
    print('Spike extraction: ')
    for i,electrode in enumerate(tqdm(prova.columns)):
        ind=all_ind[i]
        channel=prova[electrode]
        cut_outs1,all_new1=cut_all(ind,channel)
        cut_outs.append(cut_outs1)
        all_new.append(all_new1)    
    # Clustering:
    final_data=[]
    print('Clustering: ')
    for channel in (tqdm(range(len(cut_outs)))):
        channel_clusters1=clus(cut_outs[channel],all_new[channel],prova.iloc[:,channel])
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
    #np.savetxt("/Users/Gaia_1/Desktop/tesi/Data after SS/%s.txt" % save_data,adj_neur, delimiter=', ', fmt='%12.8f')
    np.savetxt(f"{output_path}/{save_data}.txt", adj_neur, delimiter=', ', fmt='%12.8f')

    print('saved: ',save_data)
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
    for neuron in tqdm(list_neurons):
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
    #ks_2samp(lista_samples,ISI_healthy,mode = 'asymp')
    return dataframe

def cut_all(all,data):
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
    all_new=[]
    for i in all:
        if (i-prima >= 0) and (i+dopo <= dim):
            spike= data[(int(i)-prima):(int(i)+dopo)].squeeze()
            media=(np.mean(spike))
            std=np.std(spike)
            spike_std=(spike-media)/std
            if abs(std)<=coeff*abs(signal_std):
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

################################POINT PROCESS
def Bayesian_mixture_model(ISI_data):
    import scipy.stats as st
    with pm.Model() as model:
        ##### WALD DISTRIBUTION (INVERSE GAUSSIAN)
        mu1 = pm.Uniform('mu1',lower=0.001,upper=0.02)
        lam1 = pm.Uniform('lam1',lower=0.001,upper=0.04)
        obs1 = pm.Wald.dist(mu=mu1,lam=lam1)


        mu2 = pm.Uniform('mu2',lower=0.02,upper=0.04)
        sigma2 = pm.Uniform('sigma2',lower=0.01,upper=0.7)
        obs2 = pm.TruncatedNormal.dist(mu=mu2, sigma=sigma2, lower=0.0)

        mu3 = pm.Uniform('mu3',lower=0.04,upper=0.6)
        sigma3 = pm.Uniform('sigma3',lower=0.01,upper=0.7)
        obs3 = pm.TruncatedNormal.dist(mu=mu3, sigma=sigma3, lower=0.0)
        
        w = pm.Dirichlet('w', a=np.array([1., 1., 1.]))
        like = pm.Mixture('like', w=w, comp_dists = [obs1, obs2, obs3], observed=ISI_data)

        '''
        if counter==9:
            step = pm.NUTS(target_accept=0.9)
            trace = pm.sample(step=step,draws=1000,chains=1,tune=1000,cores=4)
            ppc_trace = pm.sample_posterior_predictive(trace,model=model)
            bins = np.arange(0, .5, 1e-3) 
            plt.figure (figsize=(14,10))
            hist = np.histogram(ppc_trace['posterior_predictive']['like'].values,bins=bins)
            #plt.axis([-0.01,0.13,0,160])
            a= plt.hist(ISI_data,bins)
            plt.plot(hist[1][:-1],hist[0]/1000,linewidth=3)
            plt.show()
        '''
        
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
    return map_estimate
#######################
