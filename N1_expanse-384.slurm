#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.384.1800.100000.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=128
#SBATCH --mem=128G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu/0.15.4
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm
srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 4 -y 96
srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 4 -y 96 -k

srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 8 -y 48
srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 8 -y 48 -k

srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 12 -y 32
srun --mpi=pmi2 -n 384 ./apf -n 1800 -i 100000 -x 12 -y 32 -k
