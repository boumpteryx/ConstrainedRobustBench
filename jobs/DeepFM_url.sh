#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "DeepFM_url"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt
python autoattack\examples\eval.py --config config/url.yml --norm L2 --use_constraints 0 --model_name DeepFM
