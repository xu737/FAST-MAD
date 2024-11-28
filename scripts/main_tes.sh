current_datetime=$(date "+%Y-%m-%d %H:%M")

echo "start time: $current_datetime"

echo '------NO MASK MATRIX----------'
# train_path='/home/data/xrh/FL/AD_FL/checkpoints/h_msl_checkpoint1global_0.3.pth'
for seleck_k in 3  #6 8
do

  for periods_num in  6  #8  10
  do
    for param_loss_coef in  0.1

    do
        # Execute the Python script with the specified parameters
        echo 'periods_num $periods_num'
        python -u run.py \
        --task_name anomaly_detection \
        --is_training 1 \
        --root_path /home/data/xrh/FL/AD_FL/dataset/ \
        --model_id SMD \
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
        --dataset_to_layers '{"SMD": 1, "MSL": 2, "PSM": 3, "SWAT": 3, "UCR": 3, "IOps":3}' \
        --main_layer '{"SMD": 10, "MSL": 10, "PSM": 10, "SWAT": 0, "IOps":10, "UCR": 10}' \
        --param_loss_coef $param_loss_coef \
        --periods_num $periods_num \
        --dataset_device '{"UCR": 2, "SMD": 2, "MSL": 2, "PSM": 2, "SWAT": 2, "IOps":3}' \
        --seleck_k $seleck_k


        echo ""
        print_separator
        echo ""

    done
  done
done
current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "Finish time: $current_datetime"
#

        # --dataset_device '{"UCR": 1, "SMD": 1, "MSL": 1, "PSM": 1, "SWAT": 1, "IOps":3}'
