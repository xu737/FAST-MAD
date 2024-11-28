# Execute the Python script with the specified parameters
echo 'periods_num $periods_num'
python -u run.py \
--task_name anomaly_detection \
--is_training 1 \
--root_path dataset/ \
--model GPT4TS \
--seq_len 100 \
--d_model 1280 \
--d_ff 1280 \
--enc_in 38 \
--c_out 38 \
--train_epochs 2 \
--local_bs 64 \
--local_epoch 1 \
--mask_ratio 0.2 \
--gpu 1 \
--dataset_names '[ "UCR", "SMD", "MSL","PSM", "SWAT"]' \
--dataset_device '{"UCR": 2, "SMD": 2, "MSL": 2, "PSM": 2, "SWAT": 2, "IOps":3}' \
