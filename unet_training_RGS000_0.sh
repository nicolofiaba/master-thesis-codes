#!/bin/sh
#SBATCH --reservation=prj-elsa ## Use the node reserved for CAN project
#SBATCH --constraint=gpu
#SBATCH --job-name=unet_RGS_000_0
#SBATCH --output=unet_RGS_000_0.log
#SBATCH --time=15:00:00
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G     ## ram per cpu (to be tuned)
#SBATCH --mail-type=ALL      ## send a message when the job start and end
#SBATCH --mail-user="nicolo.fiaba@unibo.it"  ## email address for messages

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

python Codes/UNet_0_RGS000.py 

wait