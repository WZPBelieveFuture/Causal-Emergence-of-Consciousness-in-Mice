#!/bin/sh                                      
folder_name=(des) # [des,sev] 
network_name=(16) # [16,34,52,73,74,115] 
stage_name=(1)  # top      
for fn in ${folder_name[*]}; do 
    for net_name in ${network_name[*]}; do 
        for stage_id in ${stage_name[*]}; do 
            nohup python NIS.py --cuda 10 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 0 --version 1 --epoch 1000001 &
            nohup python NIS.py --cuda 6 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 1 --version 1 --epoch 1000001 &
            nohup python NIS.py --cuda 7 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 2 --version 1 --epoch 1000001 &
            nohup python NIS.py --cuda 12 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 3 --version 1 --epoch 1000001 &
            nohup python NIS.py --cuda 9 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 4 --version 1 --epoch 1000001 &
            nohup python NIS.py --cuda 13 --stage stage${stage_id} --folder_name ${fn} --mice_id ${net_name} --data_name data --weight_id 5 --version 1 --epoch 1000001 &
        done 
    done    
done 





