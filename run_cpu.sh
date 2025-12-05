#!/bin/sh
#SBATCH --reservation=prj-elsa ## Use the node reserved for CAN project
##SBATCH --constraint=gpu
#SBATCH --job-name=jup_CPU
#SBATCH --output=jupyter.log
#SBATCH --time=8:00:00
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G     ## ram per cpu (to be tuned)
#SBATCH --mail-type=ALL      ## send a message when the job start and end
#SBATCH --mail-user="nicolo.fiaba@unibo.it"  ## email address for messages

# Navigate to working directory
cd /home/PERSONALE/nicolo.fiaba/

HOST=$(hostname)
PORT=9595

echo "Job running on node: $HOST"

source miniconda3/bin/activate
conda activate tf

# Debugging
which python
python --version
python -c "import tensorflow; print(myenv_gpu.cuda.is_available())"

jupyter notebook --no-browser --ip=$HOST --port=$PORT &> jupyter.log

wait
