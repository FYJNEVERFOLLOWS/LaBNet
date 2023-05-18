#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N ProStage2

# resource requesting, e.g. for gpu use
#$ -l h=gpu07

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh.

echo `hostname`

gpuid=1
echo "gpuid: ${gpuid}"

CUDA_VISIBLE_DEVICES=1 \
/Work18/2020/lijunjie/anaconda3/envs/torch1.8/bin/python train/train.py \
--batch-size 4 \
--n_avb_mics 2 \
--exp-dir "/Work21/2021/fuyanjie/pycode/LaBNetPro/exp/exp0210-tri2" \
--tr-clean "/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/train-clean-100_0130.lst" \
--cv-clean "/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/dev-clean_0130.lst" \
--alpha 1 \
--beta 10 
#| tee -a "ProStage1.log"

# gpuid=2
# batch_size=4
# echo "gpuid: ${gpuid}"
# CUDA_VISIBLE_DEVICES=$gpuid \
# /Work18/2020/lijunjie/anaconda3/envs/torch1.8/bin/python train/train.py \
# --num-gpu 1 \
# --batch-size $batch_size \
# --exp-dir "/Work21/2021/fuyanjie/pycode/LaBNet/exp/exp0117_4mics" \
# --log-dir "/Work21/2021/fuyanjie/pycode/LaBNet/exp/exp0117_4mics/log" \
# --tr-clean "/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/train-clean-100_0117.lst" \
# --cv-clean "/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/dev-clean_0117.lst" \
# --alpha 1 \
# --beta 10 | tee -a "/Work21/2021/fuyanjie/pycode/LaBNet/exp/exp0117_4mics/log/stage2.txt"

echo "job end time:`date`"
