![logo_polimi](https://github.com/vettorigaia/MastersThesis/assets/150171386/84b71717-f285-497f-8b95-834187778b17)

# Scope
This project was developed as my Master's Thesis in Biomedical Engineering at Politecnico di Milano.
The project aims to investigate the neural activity patterns of neurons carrying the G2019S mutation of the LRRK2 gene, which is associated with Parkinson's disease (PD).
Using Microelectrode Array (MEA) technology, neural activity was recorded from both mutated and healthy cortical networks before and after Kainic Acid stimulation, which induces neurotoxic effects. The study developed a three-phase algorithm to process raw MEA recordings (in .h5 format), to classify neural activity into healthy and mutated patterns, and then into four classes (mutated vs. control, pre- vs. post-stimulation), with high accuracy achieved through machine learning models. The research provides insights into the pathophysiological mechanisms underlying PD and offers a novel approach for studying neurodegenerative disorders at the cellular level.

# Analysis of Neural Activity Changes Associated with LRRK2 Gene Mutation: A Computational Approach using In-Vitro Networks
## Thesis Abstract
Parkinson’s disease (PD) is a debilitating neurodegenerative disorder whose etiology is strongly linked to the G2019S mutation of the LRRK2 gene - or Dardarin. 
Although this causal relationship is well established, the underlying mechanisms are not fully understood. In this thesis, we use Microelectrode Array (MEA) technology to analyze in-vitro neural networks, in order to delineate the characteristic neural activity patterns of carriers of the G2019S mutation of the LRRK2 gene, associated with PD pathogenesis.

This study utilizes recordings from both G2019S-mutated cortical neural networks and from healthy control networks in two phases: before (baseline) and after Kainic Acid stimulation. The chemical stimulation process aims to induce neurotoxic effects, and by incorporating post-stimulation data into the analysis, to record changes in neural activity.

The work focuses on the development of a three-phase algorithm capable of receiving raw MEA recordings and recognizing the population to which they belong, distinguishing between healthy and mutated activity. Initially, Spike Sorting techniques are used to identify individual neuronal activities. Subsequently, a Point Process Modeling technique is employed to characterize the parameters of the ISI curve distribution, through Maximum A Posteriori Estimation. Finally, a classification phase is performed, where the model is trained with the parameters obtained from the modeling, and the data are categorized initially into mutated and control neurons, and subsequently, in a multi-class problem, both in terms of type of population and baseline vs. post-stimulation activity.

The results obtained from binary classification demonstrate an excellent accuracy of 99% (mutated vs. control) using a Support Vector Machine model.

Furthermore, in a more nuanced four-class classification scenario (mutated vs. control and pre- vs. post-stimulation), a Random Forest classifier achieves an accuracy of 77%, showcasing its efficacy in identifying both the genetic mutation and the stimulation-induced changes in neural activity.

The research sheds light on the complex interplay between the LRRK2 mutation and neuronal activity dynamics, providing valuable insights into the pathophysiological mechanisms underlying Parkinson’s disease, as confirmed by the obtained results. Moreover, the computational framework developed in this study offers a novel approach for studying neurodegenerative disorders at the cellular level, with implications for future diagnostic and therapeutic strategies.

## The Data
The data utilized in this study comprises recordings from 12 neural networks of cortical in-vitro neurons, half of which carry the genetic mutation of the LRRK2 gene. To induce pluripotency, Episomal iPSC reprogramming vectors are employed on dermal fibroblasts using Yamanaka factors. Subsequently, the cells are cultured and expanded on poly-L-Ornithine and laminin, followed by placement in a standard humidified air incubator for neural stem cell growth. Synchronized differentiation into cortical neurons occurs over 15 days using NSC reagent bundles and media. Recordings are obtained from resulting cultures, starting from the baseline recordings, circa 400 seconds long. Half of each neuron population undergo chemical stimulation with Kainic Acid, applied for 30 minutes, followed by recordings 24 hours post-stimulation. The chemical component is meant to provoke neuro-toxicity effects. The electrical activities were recorded through the use of Multi-Electrode Array (MEA) technology with 3-nodal microfluidic chips, specifically customized. Each nodal chamber, with an octagonal shape of 4mm in diameter, accommodates neurons for growth and development. The chambers are interconnected via micro-tunnels (104 in total), including axons and dendrites from neighboring neurons. Beneath the microfluidic chips, 59 micro-electrodes record the neuronal activity, alongside a reference electrode. The resulting files are converted to .h5 format for computational processing.

## Spike Sorting
The spike sorting process consists of processing raw MEA data to extract spikes and assign them to individual neurons, thereby defining each neuron’s complete activity.

Initially, the signal undergoes filtering with a band-pass Butterworth filter ranging from 300 to 3000Hz, followed by the application of a notch filter at critical frequencies on the Reference channel, which is subsequently subtracted from every other channel. To discern spikes from background noise, a threshold is set at 3 times the Median Absolute Deviation (MAD) of the signal over one-minute windows. After initial peak detection, each spike is extracted as a 3ms window around the peak index and evaluated to determine if it meets the criteria for a real spike. These criteria primarily involve mean and standard deviation considerations, as noisy and non-physiological spikes tend to exhibit high values in these metrics. Once the spikes are selected, they undergo standardization and the first three Principal Components, obtained through PCA, are clustered using the k-means clustering method. The cluster composition with the highest silhouette score is chosen. If no cluster configuration achieves a silhouette score above 0.5 (the standard threshold), a single cluster is defined.

## Point Process Modeling
Point Process Modeling is a method used to characterize the inter-spike intervals (ISIs) of neurons, reflecting their firing activity characteristics. In ideal Integrate-and-Fire models, neuronal firing activity follows a Brownian motion pattern, making the inverse Gaussian distribution the most suitable for modeling the neurons’ ISIs. In our study, which examines both healthy neuron populations and those carrying a genetic mutation, the Dirichlet Mixture model emerged as the most effective. This model comprises one inverse Gaussian and two Gaussian distributions. The inverse Gaussian and the first Gaussian distributions are focused on lower time difference values to capture the main peak of ISIs, while the second Gaussian distribution accounts for the tail. The ranges of the parameters (prior probability) are set as follows:

• Inverse Gaussian mean: 0-0.2

• Inverse Gaussian lambda: 0.0001-0.1

• 1st Gaussian mean: 0-0.2

• 1st Gaussian sigma: 0.01-0.7

• 2nd Gaussian mean: 0.2-0.6

• 2nd Gaussian sigma: 0.01-0.7

The model for each neuron is chosen using Maximum A Posteriori estimation, based on Bayes’ theorem. Bayes’ theorem, denoted as:


P(θ|y)=[P(y|θ)P(θ)]/P(y)


represents the relationship between the posterior probability P(θ|y), the likelihood function P(y|θ), and the prior probability P(θ) of the parameters θ. In our context, we define the prior probabilities of parameters based on their predetermined ranges, as listed above, while the likelihood function is derived from the inter-spike intervals data. By employing MAP estimation, we aim to identify the parameter values that maximize the posterior probability distribution P(θ|y). Each model’s performance is then evaluated using KS distance metrics and KS plots.

## Classification
The final stage of our processing pipeline encompasses classification tasks, which include a binary classification and a 4-class problem. In the binary classification, networks are categorized into control and LRRK2-mutated populations. Meanwhile, the 4-class problem distinguishes between control vs. mutated networks and baseline vs. post-stimulation activity. Transitioning from binary to multi-class classification impacts certain classifiers differently. While decision tree and random forest classifiers seamlessly adapt to the new framework, logistic regression and support vector machine classifiers utilize strategies such as one vs. rest or one vs. all. Through our evaluation process, we identified the most effective classifiers, by evaluating several performance metrics and choosing the one with the highest scores. When two classifiers both had similarly high performances, the one with the highest average score, computed by averaging all the metrics, was selected.

## Results
The key points of the obtained results, discussed in the thesis are:

• The control population closely aligns with an almost “pure” Inverse Gaussian model (Table 1), consistent with the theoretical framework outlined in Section 2.8 in the thesis.

• Post-stimulation activity demonstrates divergent trajectories in the two populations (Figure 2), suggesting a potential disruption in neuron functioning due to the genetic mutation.


![Schermata 2024-06-03 alle 16 54 37](https://github.com/vettorigaia/tesi/assets/150171386/1252eeb9-6c5c-4de0-a937-c38ea618c46e)

![Schermata 2024-06-03 alle 16 55 23](https://github.com/vettorigaia/tesi/assets/150171386/d814d726-e0d8-40b9-8267-1045fbb7a9b2)

The study presented builds upon existing neuronal modeling work, aiming to refine the existing framework by integrating signal processing techniques, Bayesian classification, and machine learning methods into a comprehensive pipeline for processing, modeling, and classifying raw MEA data into 2 and 4 classes. While the focus has been on distinguishing control versus G2019S LRRK2 genetic mutation samples, the pipeline has broader applicability. 

Key improvements and advancements employed in this work include enhancing the Spike Sorting algorithm through targeted refinements, improving Point Process Modeling by fine-tuning parameter distributions, and refining the Classification phase outcomes. Although our comprehensive pipeline yielded promising results, areas for enhancement remain, particularly in spike sorting algorithms. Opportunities for refinement exist in exploring alternative classification scenarios beyond the current focus of control versus LRRK2-mutated samples.

This study leveraged advanced neurotechnology and computational methodologies to dissect the underlying mechanisms of neurodegeneration associated with the G2019S mutation in the LRRK2 gene. Through characterizing aberrant neuronal activity patterns, we shed light on potential pathophysiological pathways implicated in Parkinson’s Disease progression.

The reported differences between the healthy population and the mutated one, correlate with expected physiological considerations. As reported in Section 2.5 in the thesis, the LRRK2 gene is strictly connected to the neuron’s well-being and functional behaviors, and its mutation predisposes it to degeneration and malfunctions, observed especially after the KA stimulation, which is meant to provoke neuro-toxicity. 
While achieving its primary objective of sorting, modeling, and classifying, with excellent performance results and providing a complete computational pipeline for neuronal data analysis, this work also contributes to a deeper understanding of neuronal dynamics at a physiological level, crucial in the field of neurodegenerative disorders such as Parkinson’s Disease.
