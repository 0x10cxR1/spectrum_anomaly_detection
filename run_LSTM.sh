#!/bin/bash

python LSTM.py \
	--cmd train \
	--training_path /mnt/data/zhijing/spectrum_anomaly/campus/880M_train.txt \
	--validation_path /mnt/data/zhijing/spectrum_anomaly/campus/880M_val.txt \
	--testing_path /mnt/data/zhijing/spectrum_anomaly/campus/880M_test.txt \
	--oldmodel_path /home/zhijing/spectrum_anomaly/LSTM/result/LTE_880M.h5 \
	--oldmodel_weight_path /home/zhijing/spectrum_anomaly/LSTM/result/LTE_880M_weights.h5 \
	--model_path /home/zhijing/spectrum_anomaly/LSTM/result/new_model.h5 \
	--weight_path /home/zhijing/spectrum_anomaly/LSTM/result/new_weights.h5 \
	--batch_size 64 \
	--timesteps 100 \
	--predict_steps 25 \
	--data_dim 128 \
	--epochs 10 \
	--hidden_size 64 \
	--train_num 500000 \
	--valid_num 5000 \
	--test_num 10000 \
	--testing_res /home/zhijing/spectrum_anomaly/LSTM/result/test_res.txt \
	--testing_res_CDF /home/zhijing/spectrum_anomaly/LSTM/result/test_res_CDF.txt \