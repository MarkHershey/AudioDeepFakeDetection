


CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 256 --lr 1e-3 --run_name lr1e-3  --experiment_1

CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 256 --lr 1e-3 --run_name lr1e-3  --experiment_1 --down_sample_rate 0.1


CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-4_raw_exp1  --experiment_1 --use_rawnet

CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-4_raw_exp2  --experiment_2 --down_sample_rate 0.1 --use_rawnet










CUDA_VISBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 256 --lr 1e-2 --run_name layer_2_lr1e-2 --experiment_1

CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 256 --lr 1e-2 --run_name layer_2_lr1e-2_exp2 --experiment_2






CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp1  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1  --experiment_1





CUDA_VISBLE_DEVICES=0 python train.py --RNN_layers 1 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp1  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 3230 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1  --experiment_1






CUDA_VISBILE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp1_1000  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1_1000  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1_1000  --experiment_1



CUDA_VISBILE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp2_1000  --experiment_2


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1_1000  --experiment_2


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 1000 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1_1000  --experiment_2















CUDA_VISBILE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp1_500  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1_500  --experiment_1


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1_500  --experiment_1



CUDA_VISBILE_DEVICES=1 python train.py --RNN_layers 1 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer1_exp2_500  --experiment_2


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 2 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer2_exp1_500  --experiment_2


CUDA_VISIBLE_DEVICES=1 python train.py --RNN_layers 3 --RNN_hidden 500 --batch_size 8 --lr 1e-4 --run_name lr1e-3_layer3_exp1_500  --experiment_2