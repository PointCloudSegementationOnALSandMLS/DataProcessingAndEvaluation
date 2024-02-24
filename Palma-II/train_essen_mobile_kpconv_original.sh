#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --partition=gputitanrtx
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=6

#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_jaku01@uni-muenster.de

ml purge
ml load palma/2019b  GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4
ml load Python/3.7.4

cd KPConv-PyTorch
cd cpp_wrappers
sh compile_wrappers.sh
cd ..
python train_Essen.py > filename.txt
