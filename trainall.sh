#CUDA_VISIBLE_DEVICES=1
# LCLD
python train.py --config config/lcld_v2_time.yml --model_name DeepFM --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name VIME --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024

# Malware
python train.py --config config/malware.yml --model_name DeepFM --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name VIME --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "constrained-robustbench" --api_key "tJX8KSgUh11CrvELlxhNht230"
