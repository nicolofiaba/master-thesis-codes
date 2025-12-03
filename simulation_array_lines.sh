#!/bin/bash
#SBATCH --reservation=prj-elsa
#SBATCH --job-name=sim_gelsa_array
#SBATCH --output=logs/sim_gelsa_%A_%a.log
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user="nicolo.fiaba@unibo.it"
#SBATCH --array=0-23

cd /home/PERSONALE/nicolo.fiaba/

HOST=$(hostname)
echo "Job running on node: $HOST"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

source miniconda3/bin/activate
conda activate tf

which python
python --version

PT_IDS=(
  30454 30378 30370 30358 30362 30458 
  30512 30364 30372 30488 30456 30432 
  30381 30473 30365 30369 30449 30509 
  30375 30487 30379 30507 30371 30399
)

PT_ID=${PT_IDS[$SLURM_ARRAY_TASK_ID]}

# Assign Grism+Tilt combination

TASK_ID=$SLURM_ARRAY_TASK_ID

if [ $TASK_ID -lt 6 ]; then
    GRISM="RGS000_0"
elif [ $TASK_ID -lt 12 ]; then
    GRISM="RGS000_minus4"
elif [ $TASK_ID -lt 18 ]; then
    GRISM="RGS180_0"
else
    GRISM="RGS180_4"
fi

echo "Using pt_id: $PT_ID"
echo "Using grism: $GRISM"

python codes/simulation_full_dataset.py $PT_ID --grism=$GRISM --lines &> logs/sim_${PT_ID}_${GRISM}.txt

wait