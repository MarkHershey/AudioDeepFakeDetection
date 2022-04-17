import torch
import argparse
from pathlib import Path
from utils import set_seed
from dataset import WavDataset
from dataset import get_dataset
from RNN import RNNClassification, SimpleRNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score , roc_curve
import os
from contextlib import redirect_stdout
import logging
import numpy as np
from rawnet2 import RawNet
from copy import deepcopy
LOGGER = logging.getLogger()


def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def eval(model,data_loader, seq_length):
    model.eval()
    predictions = []
    labels = []
    for x, rate, y in tqdm(data_loader, desc="Evaluating:"):
        x = x.cuda()
        B, D = x.size()
        
        x = x.reshape(B, seq_length, int(D/seq_length))
        logit = model(x).squeeze(-1)
        
        pred =((logit > 0.5) * 1).cpu().tolist()
        gold = y.tolist()
        predictions += pred
        labels += gold
    f1 = f1_score(labels, predictions)

    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    # Compute the Equal Error Rate (EER)
    fnr = 1 - tpr
    # Ref: https://stackoverflow.com/a/46026962
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
   
    print("f1:%f",  round(f1*100, 2))
    print("eer:%f",  round(eer*100, 2))
    LOGGER.info("f1:%f"%round(f1*100, 2))
    LOGGER.info("eer:%f"%round(eer*100, 2))
    return f1


def train(args, model, train_loader, val_loader, patience, loss_func, optimizer):
    num_epoch_with_no_progress = 0
    current_epoch = 0
    current_best = 0
    seq_length = args.seq_length
    while num_epoch_with_no_progress < patience:
        model.train()
        total_loss = 0
        for i , (x, rate, y) in enumerate(tqdm(train_loader, desc="Training epoch" + str(current_epoch) +":")):

            x = x.cuda()
            B, D = x.size()
           
            x = x.reshape(B, seq_length, int(D/seq_length))
           
            y = y.type(torch.float32).cuda()
            logit = model(x).squeeze(-1)
        
            loss = loss_func(logit, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 500 == 0:
                print("epoch %d instance %d loss: %f", current_epoch,i, total_loss/500)
                LOGGER.info("epoch %d instance %d loss: %f"%(current_epoch,i, total_loss/500))
                total_loss = 0
                
            
        current_epoch += 1
        f1 = eval(model, val_loader, seq_length)
        if f1 > current_best:
            current_best = f1
            num_epoch_with_no_progress = 0
            torch.save(model.state_dict(), args.save_dir + args.run_name + "/" + "epoch_" + str(current_epoch)+ "_f1:" + str(round(f1*100, 2)) + ".pt")
        else: num_epoch_with_no_progress += 1
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_dir", type=str, default="../generated_audio")
    parser.add_argument("--real_dir", type=str, default="../LJSpeech-1.1")
    parser.add_argument("--train_val_test_split", nargs='+', type=int, default=[0.6, 0.2, 0.2])
    parser.add_argument("--seq_length", type=int, default=10, help="shard video clips into how many peices and feed into RNN")
    parser.add_argument("--RNN_layers", type=int, default=1)
    parser.add_argument("--RNN_hidden", type=int, default=6460)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5, help="will stop training if val accuracy does not increase for --patience epochs")
    parser.add_argument("--save_dir", type=str, default="./experiments/")
    parser.add_argument("--run_name", type=str, default="RNN/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--down_sample_rate", help="sampel a subset to finetune network parameters", default=1.0, type=float)
    parser.add_argument("--lr", type=float, default=1e-3, )
    parser.add_argument("--experiment_1", action="store_true")
    parser.add_argument("--experiment_2", action="store_true")
    parser.add_argument("--use_rawnet", action="store_true")

    
    args = parser.parse_args()
    set_seed(42)
    assert sum(args.train_val_test_split) == 1
    os.makedirs( args.save_dir + args.run_name , exist_ok=True)
    if os.path.exists(args.save_dir + args.run_name + "/log.txt"):
        os.remove(args.save_dir + args.run_name + "/log.txt")
    init_logger(args.save_dir + args.run_name + "/log.txt")
    fake_dirs = []
    for path in Path(args.fake_dir).iterdir():
        if path.is_dir():
            if "jsut" in str(path) or "conformer" in str(path):
                continue
            if args.experiment_1:
            
                if "ljspeech_melgan" in str(path):
                    print("load %s only", str(path))
                    fake_dirs.append(path.absolute())
                    break
            if args.experiment_2:
                if "ljspeech_melgan" not in str(path):
                    print("load %s", str(path))
                    fake_dirs.append(path.absolute())
            
    train_dataset, test_dataset = get_dataset(fake_dirs, args.real_dir, args.train_val_test_split, debug=args.debug, down_sample_rate=args.down_sample_rate, experiment_1 = args.experiment_1, experiment_2 = args.experiment_2)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                     

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    
    RNN_input_dim = 64600 // args.seq_length
    if args.use_rawnet:
        RAW_NET_CONFIG = {
        "nb_samp": 64600,
        "first_conv": 1024,   # no. of filter coefficients
        "in_channels": 1,  # no. of filters channel in residual blocks
        "filts": [20, [20, 20], [20, 128], [128, 128]],
        "blocks": [2, 4],
        "nb_fc_node": 1024,
        "gru_node": 1024,
        "nb_gru_layer": 3,
        "nb_classes": 1,
        }
        model = RawNet(deepcopy(RAW_NET_CONFIG), "cuda").to("cuda")
    else:
        model = RNNClassification(num_class=1,input_size=RNN_input_dim, hidden_size=args.RNN_hidden, num_layers=args.RNN_layers,bias=True, output_size=args.RNN_hidden//2, activation='tanh')
    model = model.cuda()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, train_loader, test_loader, args.patience, loss_func, optimizer)
    print("========================================================================")
    print("Test set result:")
    LOGGER.info("Test Result: ========================================================================")
    eval(model, test_loader, args.seq_length)
    
if __name__ == "__main__":
    main()