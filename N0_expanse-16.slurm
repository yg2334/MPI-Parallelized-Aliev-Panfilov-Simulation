#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.16.2000.100000.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
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
module load tau

#srun --mpi=pmi2 -n 2 ./apf -n 800 -i 1000 -x 2 -y 1
srun --mpi=pmi2 -n 2 $PUB/HW/hw3/apf-ref -n 800 -i 1000 -x 2 -y 1
# srun --mpi=pmi2 -n 16 tau_exec -ebs ./apf -n 1800 -i 100000 -x 2 -y 8
# pprof
srun --mpi=pmi2 -n 2 ./apf -n 800 -i 1000 -x 1 -y 2
srun --mpi=pmi2 -n 2 ./apf -n 800 -i 1000 -x 2 -y 1

# srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 16 -y 1 -k


# srun --mpi=pmi2 -n 1 ./apf -n 800 -i 2000 -x 1 -y 1
# srun --mpi=pmi2 -n 1 $PUB/HW/hw3/apf-ref -n 800 -i 2000 -x 1 -y 1

# srun --mpi=pmi2 -n 2 ./apf -n 800 -i 2000 -x 2 -y 1
# srun --mpi=pmi2 -n 2 ./apf -n 800 -i 2000 -x 2 -y 1 -k
# srun --mpi=pmi2 -n 2 ./apf -n 800 -i 2000 -x 1 -y 2
# srun --mpi=pmi2 -n 2 ./apf -n 800 -i 2000 -x 1 -y 2 -k

# # srun --mpi=pmi2 -n 2 $PUB/HW/hw3/apf-ref -n 800 -i 1000 -x 1 -y 2
# # srun --mpi=pmi2 -n 2 $PUB/HW/hw3/apf-ref -n 800 -i 1000 -x 1 -y 2 -k

# srun --mpi=pmi2 -n 4 ./apf -n 800 -i 2000 -x 4 -y 1
# srun --mpi=pmi2 -n 4 ./apf -n 800 -i 2000 -x 4 -y 1 -k

# # srun --mpi=pmi2 -n 4 ./apf -n 800 -i 1000 -x 2 -y 2
# srun --mpi=pmi2 -n 4 ./apf -n 800 -i 2000 -x 1 -y 4
# srun --mpi=pmi2 -n 4 ./apf -n 800 -i 2000 -x 1 -y 4 -k

# srun --mpi=pmi2 -n 4 $PUB/HW/hw3/apf-ref -n 800 -i 2000 -x 1 -y 4
# srun --mpi=pmi2 -n 4 $PUB/HW/hw3/apf-ref -n 800 -i 2000 -x 1 -y 4 -k
# srun --mpi=pmi2 -n 8 ./apf -n 800 -i 2000 -x 1 -y 8
# srun --mpi=pmi2 -n 8 ./apf -n 800 -i 2000 -x 1 -y 8 -k
# srun --mpi=pmi2 -n 8 ./apf -n 800 -i 2000 -x 8 -y 1
# srun --mpi=pmi2 -n 8 ./apf -n 800 -i 2000 -x 8 -y 1 -k
# srun --mpi=pmi2 -n 8 $PUB/HW/hw3/apf-ref -n 800 -i 2000 -x 1 -y 8

# srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 1 -y 16
# srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 1 -y 16 -k
# srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 16 -y 1
# srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 16 -y 1 -k
# srun --mpi=pmi2 -n 16 $PUB/HW/hw3/apf-ref -n 800 -i 2000 -x 1 -y 16


