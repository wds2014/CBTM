#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:02
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : main.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>

from sentence_transformers import SentenceTransformer
from ntms import Sawtooth, ETM, ProdLDA
from mydataset import data_loader
from model import Contrastive_Model
from trainer import Trainer_Sawtooth
import argparse
import torch


parser = argparse.ArgumentParser(description='Contranstive BERT-based Neural topic model (CBTM)')
# CBTM option
parser.add_argument('--SBERT_path', type=str, default="checkpoint/paraphrase-mpnet-base-v2", help='checkpoint checkpoints of SBERT')
parser.add_argument('--tm_type', type=str, default="etm", help='name of NTM: sawtooth|etm|prodlda')
parser.add_argument('--embedding_dim', type=int, default=100, metavar='N', help='embeddings dimension of ETMs (default: 100)')
parser.add_argument('--K', type=int, default=100, help='topic size (default: 100)')
# Training options
parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train WeTe (default: 500)')
parser.add_argument('--batchsize', type=int, default=500, help='batch size (default: 500)')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for Adam to train WeTe (default: 1e-3)')
parser.add_argument('--beta', type=float, default=0.5, help='balance coefficient for bidirectional transport cost (default: 0.5)')
parser.add_argument('--alpha', type=float, default=1.0, help='hyperparameter of contrastive loss')
parser.add_argument('--epsilon', type=float, default=1.0, help='trade-off between transport cost and likelihood (default: 1.0)')
parser.add_argument('--device', type=str, default="0", help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
parser.add_argument('--init_alpha', type=bool, default=True, help='Using K-means to initialise topic embeddings or not, setting to True will make faster and better performance.')
parser.add_argument('--train_word', type=bool, default=True, help='Finetuning word embedding or not, seting to True will make better performance.')
# Dataset options
parser.add_argument('--dataname', type=str, default='20ng_6', help='Dataset: 20ng_6|20ng_20|reuters|rcv2|web|tmn|dp')

# Other options
parser.add_argument('--glove', type=str, default="./glove.6B/glove.6B.100d.txt", help='load pretrained word embedding')
# parser.add_argument('--glove', type=str, default=None, help='load pretrained word embedding')
parser.add_argument('--every_test', type=int, default=5, help='test every test_num epoch')
parser.add_argument('--save_phi', type=int, default=10, help='save phi every save_phi epoch')
parser.add_argument('--save_path', type=str, default='CBTM_result', help='path for saving topics')

args = parser.parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
args.device = device

if __name__ == '__main__':
    SBERT = SentenceTransformer(args.SBERT_path)

    train_dataloader, voc_size, voc, train_num, p_w = data_loader(bert_model=SBERT, dataname=args.dataname, voc=None, mode='train',
                                                                  batch_size=args.batch_size, shuffle=True, drop_out=False, num_workers=4)
    test_dataloader, _, _, _, _ = data_loader(bert_model=SBERT, dataname=args.dataname, voc=None, mode='test',
                                                                batch_size=args.batch_size, shuffle=True, drop_out=False, num_workers=4)
    if args.dataname == 'r8':
        clc_num = 8
    elif args.dataname == ' 20ng':
        clc_num = 20
    elif args.dataname == "r52":
        clc_num = 52
    elif args.dataname == 'ohsumed':
        clc_num = 23
    elif args.dataname == 'agnews':
        clc_num = 4

    ##### for contextual encoder
    del SBERT

    if args.tm_type == 'sawtooth':
        tm = Sawtooth(k=[100], h=[100], v=voc_size, emb_dim=100)
    elif args.tm_type == 'etm':
        tm = ETM(num_topics=100, vocab_size=voc_size, t_hidden_size=100, rho_size=100, emsize=100,
                 enc_drop=0.5, device=device)
    elif args.tm_type == 'prodlda':
        tm = ProdLDA(num_topic=100, voc_size=voc_size, hidden_sizes=(100, 100), dropout=0.2, device=device)
    else:
        print(f'unknown NTM type')

    CTM = Contrastive_Model(topic_model=tm, device=args.device)

    trainer = Trainer_Sawtooth(model=CTM, dataname=dataname, tm_name=tm_name, contrastive_alpha=contrastive_alpha,
                               p_w=p_w, train_dataloader=train_dataloader, test_dataloader=test_dataloader, voc=voc,
                               n_epoch=1000, sample_k=15, class_num=clc_num, test_num=50, train_split=train_num,
                               device=device)
    trainer.train()