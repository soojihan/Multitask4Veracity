# Simplified Multitask4Veracity
This repository contains modified code for the paper "All-in-one: Multi-task Learning for Rumour Stance classification,Detection and Verification" by E. Kochkina, M. Liakata, A. Zubiaga 

See Kochkina's repository for more details, access via https://github.com/kochkinaelena/Multitask4Veracity  

This source code provides reproducebility to our data augmentation paper as following:

Han S., Gao, J., Ciravegna, F. (2019). "Neural Language Model Based Training Data Augmentation for Weakly Supervised Early Rumor Detection", The 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2019), Vancouver, Canada, 27-30 August, 2019

Han S., Gao, J., Ciravegna, F. (2019). "Data Augmentation for Rumor Detection Using Context-Sensitive Neural Language Model With Large-Scale Credibility Corpus", Seventh International Conference on Learning Representations (ICLR) LLD,New Orleans, Louisiana, US 

# How to Run

\<to be updated\>

# Dataset

We share our LOOCV dataset used in the paper and raw augmented rumour dataset, which can be used to reproduce our results for your interest.

## LOOCV Development set and Test set


## Raw tweets rumour corpus

Augmented Raw data is based on [PHEME 6392078 dataset](https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) and can be downloaded at 
https://figshare.com/account/projects/63092/articles/8066759 (still underreview by figshare and will be available soon; it is also available upon request)

## ELMo model
  
Credbank Fine-tuned ELMo(ELMo_CREDBANK) used in LLD paper can be downloaded via http://staffwww.dcs.shef.ac.uk/people/J.Gao/data/elmo_credbank/elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_12262018.hdf5

If you use the version from this Git repository or our augmented data (BostonBombing-Aug v1.0), please cite: 

Han S., Gao, J., Ciravegna, F. (2019). "Data Augmentation for Rumor Detection Using Context-Sensitive Neural Language Model With Large-Scale Credibility Corpus", *Seventh International Conference on Learning Representations (ICLR) LLD*, May 2019, New Orleans, Louisiana, US
