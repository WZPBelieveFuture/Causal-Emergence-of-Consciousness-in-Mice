#!/bin/sh                                  
folder_name=(des) # [des,sev] 
network_name=(16) # [16,34,52,73,115]
for fn in ${folder_name[*]}; do 
    for net_name in ${network_name[*]}; do 
        nohup python NIS.py --cuda 11 --stage stage1 --folder_name ${fn} --mice_id ${net_name} --data_name data --learning_rate 0.0001 --scale_id 0 --train_stage 1 --epoch 1000001 &
        nohup python NIS.py --cuda 13 --stage stage5 --folder_name ${fn} --mice_id ${net_name} --data_name data --learning_rate 0.0001 --scale_id 0 --train_stage 1 --epoch 1000001 &
        nohup python NIS.py --cuda 4 --stage stage11 --folder_name ${fn} --mice_id ${net_name} --data_name data --learning_rate 0.0001 --scale_id 0 --train_stage 1 --epoch 1000001 &
    done    
done