# ADHD

Literature reasearch on potential network dysfuntions in ADHD.

(qMRI_castellanos_1996) =

## qMRI for ADHD, Castellanos, 1996

````{margin}
```{note}
Quantitative brain magnetic resonance imaging in attention-deficit hyperactivity disorder 
https://pubmed.ncbi.nlm.nih.gov/8660127/
```
````

check this first :
Cognitive Neuroscience of Attention Deficit Hyperactivity Disorder (ADHD) and Its Clinical Translation, 2018
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5884954/

check review 2005 : https://www.nmr.mgh.harvard.edu/BushLab/Site/Publications_files/Bush-2005-ADHD%20Imaging%20Review.pdf
check review 2026 : https://onlinelibrary.wiley.com/doi/10.1111/cch.12139

(phase_based_EEG)=

## Phase-based brain connectivity ADHD - EEG

````{margin}
```{note}
Potential biomarker for early detection of ADHD using phase-based brain connectivity and graph theory
https://pubmed.ncbi.nlm.nih.gov/37668834/
```
````

ADHD prevalence : 5.3% in adolencence, with 2:1 male to femail ratio
heredity plays a role

````{margin}
* See Ahmadlou et al 8,9 for EEG synchronization likelihood and fuzzy SL (FSL), accuracy 97.1 and 87.5 (adolescents)
* See Kiiski et al 13 for weithed phase lag index (adult)
* see chen et al 17 for clustering coefficient and short path length 
* Michelini et al 20 for imaginary part of coherence
````

### Results

* phase based - accuracy of 99.174%
* EEG connectivity
* biomarkers : subgraph centrality of **phase lag index** in beta and delta freq
* node betweeness centrality of **inter-site phase clustering cennectivity** in delta and thata bands

### Methods

PLI or IDPC connectivity matrix, binary thresholding
theshold : one standard deviation above median connectivity

PLI : phase angle diff :

* symmetrical dist around 0 : fake connectivity
* positive, negativ : real connectivity
* flat distribution : no connectivity

IPSP : consisency in phase angle between two signals

Measures computed from those matrices :

* CC
* local efficiency
* Louvain commmunity
* node strength
* node degree
* betweeness centrality
* subgraph centrality

Classifiers : KNN, SVM, NB, LDA, DT, ANN

check again for interpretation of channel wise features but not too impressive really

(precuneus)=

## Cingulate cortex - Precuneus interactions (2008)

This is some text.

%Here is my nifty citation {cite}`perez2011python`.