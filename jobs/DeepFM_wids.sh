#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "DeepFM_wids"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -p batch
#SBATCH --time=10:00:00
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#module load lang/Python/3.6.0

pip install --upgrade pip setuptools wheel config-utils configutils
pip install -r requirements.txt
pip install -e git+https://github.com/boumpteryx/constrained-attacks.git@bd6e8448892621198ac8b0e14c250fb035dd4f2d#egg=constrained_attacks
python autoattack/examples/eval.py --config config/wids.yml --norm L2 --use_constraints 0 --model_name DeepFM
