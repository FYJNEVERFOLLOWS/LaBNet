#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N test

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=gpu05
echo `hostname`

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh.

gpuid=2
echo "gpuid: ${gpuid}"
CUDA_VISIBLE_DEVICES=$gpuid /Work21/2021/fuyanjie/anaconda3/envs/torch1.8+cu111/bin/python test/test.py \
--batch-size 5 \
--ckpt-path "/path/to/your/checkpoint.pt" \
--tt-clean "/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/test-clean_0130.lst" \
# --write-wav True


sleep 10
echo "job end time:`date`"
