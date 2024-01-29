#!/bin/bash

#SBATCH -t 24:00:00

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_example

# MODULES
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load h5py/3.7.0-foss-2022a

# SET ENV VARIABLES
echo "--- ENVIRONMENT VARIABLES ---"
# TRAINING
FILE_PATH=$TMPDIR/datasets/mri.hdf5
SCHEDULE_SAMPLER=uniform
LR=1e-4
WEIGHT_DECAY=0.0
LR_ANNEAL_STEPS=0
BATCH_SIZE=8
MICROBATCH=-1
EMA_RATE="0.9999"
LOG_INTERVAL=50
SAVE_INTERVAL=1000
RESUME_CHECKPOINT=""
USE_FP16=false
FP16_SCALE_GROWTH=1e-3
MAX_STEPS=46875
# MAX_STEPS=10
LABELS=all

# UNET & DIFFUSION
IMAGE_SIZE=256
IN_CHANNELS=4
NUM_CHANNELS=128
NUM_RES_BLOCKS=2
NUM_HEADS=2
NUM_HEADS_UPSAMPLE=-1
ATTENTION_RESOLUTIONS="32,16,8"
DROPOUT=0.1
LEARN_SIGMA=true
CLASS_COND=true
DIFFUSION_STEPS=1000
NOISE_SCHEDULE=linear
TIMESTEP_RESPACING=""
USE_KL=false
PREDICT_XSTART=false
RESCALE_TIMESTEPS=false
RESCALE_LEARNED_SIGMAS=false
USE_CHECKPOINT=false
USE_SCALE_SHIFT_NORM=false
CHANNEL_MULT="1,1,2,3,4"
DIMENSIONS=2
SAMPLE_DISTANCE=250
NOISE_FN=simplex

ORGANS=brain


OUTPUT_DIR="$HOME"/output/"$SLURM_JOB_NAME"_"$SLURM_JOBID"
echo "-----------------------------"

# SETUP DIRECTORIES AND DATASET
echo "--- SETTING UP DATASET AND DIRECTORIES ---"
cd "$TMPDIR"

mkdir "$TMPDIR"/datasets
echo "COPY ARCHIVE"
cp -r $HOME/datasets/preprocessed/BraTS.hdf5 $TMPDIR/datasets/mri.hdf5
mkdir -p $OUTPUT_DIR
echo "------------------------------------------"

# RUN EXPERIMENT
echo "--- START EXPERIMENT ---"
start=$SECONDS
mpiexec -n 4 python $HOME/improved-diffusion/scripts/multi_mri_train.py \
	--file_path $FILE_PATH \
	--schedule_sampler $SCHEDULE_SAMPLER \
	--lr $LR \
	--weight_decay $WEIGHT_DECAY \
	--batch_size $BATCH_SIZE \
	--microbatch $MICROBATCH \
	--ema_rate $EMA_RATE \
	--log_interval $LOG_INTERVAL \
	--save_interval $SAVE_INTERVAL \
	--use_fp16 $USE_FP16 \
	--fp16_scale_growth $FP16_SCALE_GROWTH \
	--max_steps $MAX_STEPS \
	--image_size $IMAGE_SIZE \
	--in_channels $IN_CHANNELS \
	--num_channels $NUM_CHANNELS \
	--num_res_blocks $NUM_RES_BLOCKS \
	--num_heads $NUM_HEADS \
	--num_heads_upsample $NUM_HEADS_UPSAMPLE \
	--attention_resolutions $ATTENTION_RESOLUTIONS \
	--dropout $DROPOUT \
	--learn_sigma $LEARN_SIGMA \
	--class_cond $CLASS_COND \
	--diffusion_steps $DIFFUSION_STEPS \
	--noise_schedule $NOISE_SCHEDULE \
	--use_kl $USE_KL \
	--predict_xstart $PREDICT_XSTART \
	--rescale_timesteps $RESCALE_TIMESTEPS \
	--rescale_learned_sigmas $RESCALE_LEARNED_SIGMAS \
	--use_checkpoint $USE_CHECKPOINT \
	--use_scale_shift_norm $USE_SCALE_SHIFT_NORM \
	--channel_mult $CHANNEL_MULT \
	--dimensions $DIMENSIONS \
	--sample_distance $SAMPLE_DISTANCE \
	--output_dir $OUTPUT_DIR \
	--noise_fn $NOISE_FN \
	--organs $ORGANS \
	--labels $LABELS
duration=$(( SECONDS - start))
echo "experiment ran for $duration seconds."
echo "------------------------"
echo "--- DONE ---"