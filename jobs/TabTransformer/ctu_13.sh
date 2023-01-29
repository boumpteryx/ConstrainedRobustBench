#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "TabTransformer_ctu_13"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -p batch
#SBATCH --time=4:00:00
#SBATCH --mail-type=end,fail
#SBATCH  --mem-per-cpu=8G
#SBATCH --output=~/constrained-attacks/logs/%x.%j.out
#SBATCH --error=~/constrained-attacks/-%x.%j.err

##SBATCH --mem=2G

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
pip install -e git+https://github.com/serval-uni-lu/constrained-attacks.git@ee7acd54f96b974157c55bbf551d4089a9a7a4e2


## 4/255
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 0 --model_name TabTransformer --epsilon 0.0157
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 1 --model_name TabTransformer --epsilon 0.0157

## 8/255
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 0 --model_name TabTransformer --epsilon 0.0314
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 1 --model_name TabTransformer --epsilon 0.0314

## 16/255
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 0 --model_name TabTransformer --epsilon 0.0627
python autoattack/examples/eval.py --verbose 1 --config config/ctu_13_neris.yml --norm L2 --use_constraints 1 --model_name TabTransformer --epsilon 0.0627
