#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export num_workers=0
export total_hour=50
export num_classes=9
export use_pretrained='False'
export patience=20

export implementation='FlexMatch'
export normalization='unnormalized_HWC'

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='PathMNIST'
export data_seed=0
export training_seed=1
export nlabels=450 

export train_dir="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/experiments/$dataset_name/$normalization/data_seed$data_seed/Worker0N1_MultipleSeeds_50h_evaluate1h/training_seed$training_seed/RandomSearch_RealisticPlan/nlabels450_val450_RealisticPlan/$implementation"
mkdir -p $train_dir

export script="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/Semi-supervised/$dataset_name/RandomSearch_RealisticPlan/$implementation/$implementation.py"


export arch='resnet18'
export train_epoch=2000 
export start_epoch=0
export eval_every_Xepoch=1


#data paths
export l_train_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels450_val450_realistic/l_train.npy"

export u_train_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels450_val450_realistic/u_train.npy"

export val_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels450_val450_realistic/val.npy"

export test_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels450_val450_realistic/test.npy"


#shared config
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=448 #default
export em=0 #default


#PL config, candidate hypers to search
# export optimizer_type='Adam'
# export lr=0.03
# export wd=5e-4
# export lambda_u_max=1.0
# export temperature=1.0
# export mu=7
# export threshold=0.95
# export unlabeledloss_warmup_schedule_type='NoWarmup'
# export unlabeledloss_warmup_pos=0.4 #FixMatch algo did not use unlabeled loss rampup schedule
# export lr_warmup_epochs=0
# export lr_schedule_type='CosineLR'
# export lr_cycle_epochs=$train_epoch

export optimizer_type='Adam'
export temperature=1.0
export mu=7
export threshold=0.95
export unlabeledloss_warmup_schedule_type='NoWarmup'
export unlabeledloss_warmup_pos=0.4 #FixMatch algo did not use unlabeled loss rampup schedule
export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ./do_experiment.slurm
fi


