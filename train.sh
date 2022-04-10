


CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 256 --lr 1e-3 --run_name lr1e-3  --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 256 --lr 1e-2 --run_name lr1e-2 --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 256 --lr 5e-2 --run_name lr5e-2 --down_sample_rate 0.1


CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 256 --lr 1e-3 --run_name layer_2_lr1e-3 --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 256 --lr 1e-2 --run_name layer_2_lr1e-2 --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 256 --lr 5e-2 --run_name layer_2_lr5e-2 --down_sample_rate 0.1



CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 1000 --batch_size 256 --lr 1e-3 --run_name lr1e-3_hidden1000 --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 1000 --batch_size 256 --lr 1e-2 --run_name lr1e-2_hidden1000 --down_sample_rate 0.1

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 1000 --batch_size 256 --lr 5e-2 --run_name lr5e-2_hidden1000 --down_sample_rate 0.1