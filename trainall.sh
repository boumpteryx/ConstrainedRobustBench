# LCLD
python train.py --config config/lcld_v2_time.yml --model_name DeepFM --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230" --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name VIME --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024
python train.py --config config/lcld_v2_time.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"  --batch_size 512 --val_batch_size 1024

# Malware
python train.py --config config/malware.yml --model_name DeepFM --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name VIME --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/malware.yml --model_name ModelTree --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"

# URL
python train.py --config config/url.yml --model_name DeepFM --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/url.yml --model_name VIME --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/url.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"

# WIDS
python train.py --config config/wids.yml --model_name DeepFM --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/wids.yml --model_name VIME --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/wids.yml --model_name TabTransformer --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"


# CTU
python train.py --config config/ctu_13_neris.yml --model_name DeepFM --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/ctu_13_neris.yml --model_name VIME --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"
python train.py --config config/ctu_13_neris.yml --model_name TORCHRLN --optimize_hyperparameters  --workspace "yamizi" --api_key "tJX8KSgUh11CrvELlxhNht230"


