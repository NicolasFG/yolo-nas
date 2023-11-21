#!/bin/bash
#SBATCH -J tesis3 # nombre del job
#SBATCH -p investigacion # nombre de la particion
#SBATCH -c 8  # numero de cpu cores a usar
#SBATCH --nodelist=g001
#SBATCH --mem=32GB
module load gcc/9.2.0
module load cuda/12.2
module load python/3.9.18
export PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100
python3 main.py 
