# URL
python --config config/url.yml --norm L2 --use_constraints 0 --model_name Net --transfer_from Linear --version transfer
python --config config/url.yml --norm L2 --use_constraints 1 --model_name Net --transfer_from Linear --version transfer
python --config config/url.yml --norm L2 --use_constraints 0 --model_name Net --transfer_from Linear --version transfer

