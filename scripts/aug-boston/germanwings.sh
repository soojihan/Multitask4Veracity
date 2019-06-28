#!/bin/bash
#$ -P rse -l rmem=30G -l h_rt=96:00:00 -M sooji.han@sheffield.ac.uk -m beas

module load libs/CUDA/9.1.85/binary
module load libs/cudnn/7.0/binary-cuda-9.1.85
module load dev/gcc/4.9.4
module load apps/python/conda
module load apps/torch/nvidia-7/gcc-4.9.4-cuda-8.0-cudnn-5.1-conda-3.4

source activate allennlp-prj

#reset allennlp local database cache directory
#export ALLENNLP_CACHE_ROOT="/fastdata/acp16sh/.batchannotaton"

#disable hdf5 file locking (e.g., for training ELMo model) which we do not have permission in iceberg
export HDF5_USE_FILE_LOCKING=FALSE



python /mnt/fastdata/acp16sh/newbaseline/src/outer.py --model='mtl2detect' --data='pheme5' --fname='germanwings' --search=True --ntrials=20 --train_path='/mnt/fastdata/acp16sh/newbaseline/data/aug-boston/germanwings/train' --heldout_path='/mnt/fastdata/acp16sh/newbaseline/data/aug-boston/germanwings/heldout' --test_path='/mnt/fastdata/acp16sh/newbaseline/data/aug-boston/germanwings/test' --save_path='/mnt/fastdata/acp16sh/newbaseline/output/aug-boston/germanwings'

