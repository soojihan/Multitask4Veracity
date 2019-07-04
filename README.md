# Simplified Multitask4Veracity
This repository contains modified code for the paper "All-in-one: Multi-task Learning for Rumour Stance classification,Detection and Verification" by E. Kochkina, M. Liakata, A. Zubiaga 

See Kochkina's repository for more details, access via https://github.com/kochkinaelena/Multitask4Veracity  

This source code provides reproducebility to our data augmentation paper as following:

[1] Han S., Gao, J., Ciravegna, F. (2019). "Neural Language Model Based Training Data Augmentation for Weakly Supervised Early Rumor Detection", The 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2019), Vancouver, Canada, 27-30 August, 2019

[2] Han S., Gao, J., Ciravegna, F. (2019). "Data Augmentation for Rumor Detection Using Context-Sensitive Neural Language Model With Large-Scale Credibility Corpus", Seventh International Conference on Learning Representations (ICLR) LLD,New Orleans, Louisiana, US 

# How to Run

- Training and Evaluation

python src/outer.py --model='mtl2detect' --data='pheme5' --fname=\<model_output_filename\> --search=True --ntrials=30
--train_path=\<train_set_directory\> --holdout_path=\<holdout_set_directory\> --test_path=\<test_set_directory\> 
--save_path=\<model_output_directory\> --params_file=\<parameter_filename\>

- Evaluation using optimised parameters

python src/outer.py --model='mtl2detect' --data='pheme5' --fname=\<model_output_filename\>
--train_path=\<train_set_directory\> --holdout_path=\<holdout_set_directory\> --test_path=\<test_set_directory\> 
--save_path=\<model_output_directory\> --params_file=\<parameter_filename\>

# Dataset

We share our LOOCV dataset used in the paper and raw augmented rumour dataset, which can be used to reproduce our results for your interest.

## LOOCV Development set and Test set

# Trained models and results presented in [1]
We share trained models and parameters for the evaluation results presented in our ASONAM2019 paper[1]. 

## Raw tweets rumour corpus

Augmented Raw data is based on [PHEME 6392078 dataset](https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) and can be downloaded at 
https://figshare.com/account/projects/63092/articles/8066759 (still underreview by figshare and will be available soon; it is also available upon request)

## ELMo model
  
Credbank Fine-tuned ELMo(ELMo_CREDBANK) used in LLD paper can be downloaded via http://staffwww.dcs.shef.ac.uk/people/J.Gao/data/elmo_credbank/elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_12262018.hdf5

If you use the version from this Git repository or our augmented data (BostonBombing-Aug v1.0), please cite: 

Han S., Gao, J., Ciravegna, F. (2019). "Data Augmentation for Rumor Detection Using Context-Sensitive Neural Language Model With Large-Scale Credibility Corpus", *Seventh International Conference on Learning Representations (ICLR) LLD*, May 2019, New Orleans, Louisiana, US
