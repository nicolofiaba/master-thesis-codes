#!/bin/bash
#SBATCH --reservation=prj-elsa ## Use the node reserved for CAN project
#SBATCH --constraint=gpu
#SBATCH --job-name=att_unet_training
#SBATCH --output=att_unet_training.log
##SBATCH --time=15:00:00
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G     ## ram per cpu (to be tuned)
#SBATCH --mail-type=ALL      ## send a message when the job start and end
#SBATCH --mail-user="nicolo.fiaba@unibo.it"  ## email address for messages
#SBATCH --qos=long

# Navigate to working directory
cd /home/PERSONALE/nicolo.fiaba/

HOST=$(hostname)
PORT=9596

echo "Job running on node: $HOST"

source miniconda3/bin/activate
conda activate tf

# Debugging
which python
python --version

python codes/att_UNet.py --gpu=0  --lr=1e-4 --batch=16 --epochs=500 --grism="RGS180_0" &> RGS180_0_log.txt &
python codes/att_UNet.py --gpu=1  --lr=1e-4 --batch=16 --epochs=500 --grism="RGS000_0" &> RGS000_0_log.txt &
python codes/att_UNet.py --gpu=2  --lr=1e-4 --batch=16 --epochs=500 --grism="RGS000_minus4" &> RGS000_minus4_log.txt &
python codes/att_UNet.py --gpu=3  --lr=1e-4 --batch=16 --epochs=500 --grism="RGS180_4" &> RGS180_4_log.txt &

wait
