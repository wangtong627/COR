Version: 2025/04/04

Train:
accelerate launch \
 --config_file /data2/wang_tong/proj_cirseg/my_models/cirseg_model/config/train_config/a_cfg.yaml \
 /data2/wang_tong/proj_cirseg/my_models/cirseg_model/my_train_a.py \
 --config /data2/wang_tong/proj_cirseg/my_models/cirseg_model/config/train_config/train_config_m3.yaml

Validation:
accelerate launch \
 --config_file /data2/wang_tong/proj_cirseg/my_models/cirseg_model/config/vaild_config/vaild_a.yaml \
 /data2/wang_tong/proj_cirseg/my_models/cirseg_model/my_test.py \
 --config /data2/wang_tong/proj_cirseg/my_models/cirseg_model/config/vaild_config/vaild_config.yaml
