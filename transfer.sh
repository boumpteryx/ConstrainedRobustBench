# URL
python --config config/url.yml --norm L2 --use_constraints 0 --model_name Net --transfer_from Linear --version transfer
python --config config/url.yml --norm L2 --use_constraints 1 --model_name Net --transfer_from Linear --version transfer

for dataset in config/url.yml config/wids.yml config/lcld_v2_time.yml config/ctu_13_neris.yml config/malware.yml
do
  for constr in 0 1
  do
    for
    do
      for source_model in Linear Net VIME TabTransformer DeepFM TORCHRLN SAINT
      do
        for target_model in Linear Net VIME TabTransformer DeepFM TORCHRLN SAINT KNN ModelTree RandomForest DecisionTree CatBoost XGBoost LightGBM
        do
          python --config dataset --norm L2 --use_constraints constr --model_name target_model --transfer_from source_model --version transfer
        done
      done
    done
  done
done

