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

\<<To be updated. Experimental dataset will be uploaded soon. \>>

# Trained models and results presented in [1]
We share trained models and parameters for the evaluation results presented in our ASONAM2019 paper[1]. 

\<<To be updated. Trained model for reproducing our results will be uploaded soon. \>>

## Raw tweets rumour corpus

Augmented Raw data is based on [PHEME 6392078 dataset](https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078). We have two versions of augmented rumor corpus.

For version 1 used in LLD2019 paper, please downloaded it via https://zenodo.org/record/3249977

For version 2 used in ASONAM2019 paper, please downloaded it via https://zenodo.org/record/3269768

## ELMo model
  
Credbank Fine-tuned ELMo(ELMo_CREDBANK) used in LLD paper can be downloaded via figshare [shef.data.11591775.v1](https://figshare.shef.ac.uk/articles/Credibility_corpus_fine-tuned_ELMo_contextual_language_model_for_early_rumor_detection_on_social_media/11591775/1) with version "12262018.hdf5" 

Credbank Fine-tuned ELMo used in ASONAM 2019 paper can be downloaded via figshare [shef.data.11591775.v1](https://figshare.shef.ac.uk/articles/Credibility_corpus_fine-tuned_ELMo_contextual_language_model_for_early_rumor_detection_on_social_media/11591775/1) with the version "10052019.hdf5" .

## Citation

If you use the version from this Git repository or our augmented data (BostonBombing-Aug v1.0), please cite: 

Han S., Gao, J., Ciravegna, F. (2019). "Data Augmentation for Rumor Detection Using Context-Sensitive Neural Language Model With Large-Scale Credibility Corpus", *Seventh International Conference on Learning Representations (ICLR) LLD*, May 2019, New Orleans, Louisiana, US

If you use the source code or our augmented data (v2.0), please cite:

 Han S., Gao, J., Ciravegna, F. (2019). "Neural Language Model Based Training Data Augmentation for Weakly Supervised Early Rumor Detection", The 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2019), Vancouver, Canada, 27-30 August, 2019
