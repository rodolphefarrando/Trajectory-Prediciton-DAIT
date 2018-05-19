#!/bin/bash -l
#SBATCH --workdir /home/farrando
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos gpu_free
#SBATCH --account civil-459
#SBATCH --reservation civil-459-project
#SBATCH --time 12:00:00

module load gcc python cuda
source ~/venv/pytorch/bin/activate
python3 ./Trajectory-Prediciton-DAIT/model_rodolphe/LSTM_2Loss.py

