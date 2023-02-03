#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --dataset_name $dataset_name\
    --data_seed $data_seed\
    --training_seed $training_seed\
    --nlabels $nlabels\
    --arch $arch\
    --train_epoch $train_epoch\
    --nimg_per_epoch $nimg_per_epoch\
    --start_epoch $start_epoch\
    --eval_every_Xepoch $eval_every_Xepoch\
    --resume $resume\
    --train_dir $train_dir\
    --l_train_dataset_path $l_train_dataset_path\
    --u_train_dataset_path $u_train_dataset_path\
    --val_dataset_path $val_dataset_path\
    --test_dataset_path $test_dataset_path\
    --labeledtrain_batchsize $labeledtrain_batchsize\
    --unlabeledtrain_batchsize $unlabeledtrain_batchsize\
    --em $em\
    --xi $xi\
    --eps $eps\
    --lambda_u_max $lambda_u_max\
    --lr $lr\
    --wd $wd\
    --lr_warmup_epochs $lr_warmup_epochs\
    --optimizer_type $optimizer_type\
    --lr_schedule_type $lr_schedule_type\
    --lr_cycle_epochs $lr_cycle_epochs\
    --unlabeledloss_warmup_schedule_type $unlabeledloss_warmup_schedule_type\
    --unlabeledloss_warmup_pos $unlabeledloss_warmup_pos\
    