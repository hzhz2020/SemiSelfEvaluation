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
export num_classes=8
export use_pretrained='False'
export patience=20

export implementation='Supervised_RE3'
export normalization='unnormalized_HWC'

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='TissueMNIST'
export data_seed=0
export training_seed=0
export nlabels=400

export train_dir="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/experiments/$dataset_name/$normalization/data_seed$data_seed/Worker0N1_MultipleSeeds_50h_evaluate1h/training_seed$training_seed/RandomSearch_RealisticPlan/nlabels400_val400_RealisticPlan/$implementation"
mkdir -p $train_dir

export script="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/Semi-supervised/$dataset_name/RandomSearch_RealisticPlan/$implementation/$implementation.py"


export arch='resnet18'
export train_epoch=2000 
export start_epoch=0
export eval_every_Xepoch=1


#data paths
export l_train_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels400_val400_realistic/l_train.npy"


export val_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels400_val400_realistic/val.npy"

export test_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/MedMNIST/$dataset_name/$normalization/data_seed$data_seed/nlabels400_val400_realistic/test.npy"


#shared config
export labeledtrain_batchsize=64 #default


#PL config, candidate hypers to search
export optimizer_type='Adam'

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


