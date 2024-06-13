#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.192.8000.8000.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
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
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48 -k

#srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 8 -y 24
#srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 8 -y 24 -k

#srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16
#srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 12 -y 16 -k
