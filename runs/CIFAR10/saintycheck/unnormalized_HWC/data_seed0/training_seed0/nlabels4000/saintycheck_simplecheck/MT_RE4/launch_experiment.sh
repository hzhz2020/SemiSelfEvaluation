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
experiment='saintycheck'

export implementation='MT_RE4'
export normalization='unnormalized_HWC'

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='CIFAR10'
export data_seed=0
export training_seed=0
export nlabels=4000 

export train_dir="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/experiments/CIFAR10/$experiment/$normalization/data_seed$data_seed/training_seed$training_seed/nlabels$nlabels/$implementation"
mkdir -p $train_dir

export script="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/SemiSelfEvaluation/Semi-supervised/CIFAR10/$implementation/$implementation.py"


export arch='wideresnet'
export train_epoch=2000 #27962 equal to train for 1<<16 kimg in MixMatch and FixMatch repo #10416 equal to train for 500,000 iterations in Oliver et al 2018
export nimg_per_epoch=4000 #size of the labeled train set
export start_epoch=0
export eval_every_Xepoch=10


#data paths
export l_train_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/$dataset_name/Regular/$normalization/data_seed$data_seed/nlabels$nlabels/l_train.npy"

export u_train_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/$dataset_name/Regular/$normalization/data_seed$data_seed/nlabels$nlabels/u_train.npy"

export val_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/$dataset_name/Regular/$normalization/data_seed$data_seed/nlabels$nlabels/val.npy"

export test_dataset_path="/cluster/tufts/hugheslab/zhuang12/SemiSelfEvaluationProject/ML_DATA/$dataset_name/Regular/$normalization/data_seed$data_seed/nlabels$nlabels/test.npy"


#shared config
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=64 #default
export em=0 #default


#PL config, candidate hypers to search
export optimizer_type='Adam'
export lr=0.03
export wd=0.0005
export lambda_u_max=50.0


export unlabeledloss_warmup_schedule_type='Linear'
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


