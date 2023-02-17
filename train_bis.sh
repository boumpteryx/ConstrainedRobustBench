# URL
python train.py --config config/url.yml --model_name DeepFM --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/url.yml --model_name VIME --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/url.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/url.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048


# WIDS
python train.py --config config/wids.yml --model_name DeepFM --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/wids.yml --model_name VIME --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/wids.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/wids.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048


# CTU
python train.py --config config/ctu_13_neris.yml --model_name DeepFM --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/ctu_13_neris.yml --model_name VIME --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/ctu_13_neris.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
python train.py --config config/ctu_13_neris.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "constrained-robustbench-minmax" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 1024 --val_batch_size 2048
