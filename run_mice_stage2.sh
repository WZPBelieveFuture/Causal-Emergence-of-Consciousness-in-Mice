#!/bin/sh                                      
folder_name=(des) # [des,sev] 
network_name=(16) # [16,34,52,73,115] 
stage_name=(1)    # [1,5,11] 
for fn in ${folder_name[*]}; do 
    for net_name in ${network_name[*]}; do 
        for stage_id in ${stage_name[*]}; do 
            nohup python NIS.py --cuda 7 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 0 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
            nohup python NIS.py --cuda 9 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 1 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
            nohup python NIS.py --cuda 6 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 2 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
            nohup python NIS.py --cuda 4 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 3 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
            nohup python NIS.py --cuda 6 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 4 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
            nohup python NIS.py --cuda 7 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --scale_id 5 --learning_rate 0.0001 --train_stage 2 --epoch 1000001 &
        done 
    done    
done 





