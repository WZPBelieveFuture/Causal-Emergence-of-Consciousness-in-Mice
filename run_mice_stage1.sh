#!/bin/sh                                  
folder_name=(des) # [des,sev] 
network_name=(16)  # [16,34,52,73,74,115]  16，34，52，73, 74
stage_name=(1) 
for fn in ${folder_name[*]}; do 
    for net_name in ${network_name[*]}; do 
        for stage_id in ${stage_name[*]}; do 
            nohup python NIS.py --cuda 12 --stage stage1 --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 0 --version 0 --epoch 1000001 &
            nohup python NIS.py --cuda 11 --stage stage5 --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 0 --version 0 --epoch 1000001 &
            nohup python NIS.py --cuda 13 --stage stage11 --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 0 --version 0 --epoch 1000001 &
        done 
    done    
done