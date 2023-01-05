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

export normalization='unnormalized_HWC'

export script="/cluster/tufts/hugheslab/zhuang12/SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/src/build_datasets/regular_ssl/$normalization/build_dataset.py"

export data_rootdir='/cluster/tufts/hugheslab/zhuang12/SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/ML_DATA/'

export seed=0
export dataset='cifar10'
export nlabels=8000

## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < /cluster/tufts/hugheslab/zhuang12/SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/build_datasets/regular_ssl/$normalization/do_BuildRegularSSL.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash /cluster/tufts/hugheslab/zhuang12/SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/src/build_datasets/regular_ssl/$normalization/do_BuildRegularSSL.slurm
fi

