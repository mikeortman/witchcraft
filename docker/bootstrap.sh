cd /vol
mkdir -p logs
tensorboard --logdir ./logs &
python $1

