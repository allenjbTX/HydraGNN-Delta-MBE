#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J dmbe_order_3
#SBATCH -o logs/order_3-%j.out
#SBATCH -e logs/order_3-%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 16

mkdir -p logs

module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/6.4.0
ml amd-mixed/6.4.0
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime

source activate /ccs/home/allenjb/HydraGNN-Installation-Frontier/hydragnn_venv
export PYTHONPATH=/ccs/home/allenjb/HydraGNN:$PYTHONPATH

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

cd /lustre/orion/scratch/allenjb/lrn070/HydraGNN-Delta-MBE/delta-mbe-models

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) \
     -c7 --gpus-per-task=1 --gpu-bind=closest \
     python train_hydragnn.py \
         --config hydra_energy_order_3.json \
         --dataset-dir serialized_datasets
