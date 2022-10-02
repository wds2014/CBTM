#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:50
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : trainer.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>


from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import os
from torch.utils.tensorboard import SummaryWriter
from data_prepare import vision_phi
from utils import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def sample_value(a, b, index):
    for i in range(len(index)):
        a[i, index[i]] = b[i, index[i]]
    return torch.clamp(a, min=0.0, max=9999.0)


def ts(x, y):
    ## calculate topic specificity as david mimno
    ## ts = 1/k * \sum_{k=1}^{K} KL(p(w|z=k) || p(w) )
    ## x: k,v
    ## y: v, a vector of word distribution in a specific corpus
    xx = torch.log(torch.from_numpy(x.T))  ## k,v
    y = np.reshape(y, (1, -1))
    yy = torch.from_numpy(y).expand_as(xx)  ## k,v
    return F.kl_div(xx, yy, reduction='batchmean')


class Trainer_Sawtooth(object):
    """
    Contrastive model Trainer
    """

    def __init__(self, model=None, dataname='20ng', tm_name='sawtooth', contrastive_alpha=1.0, p_w=None,
                 train_dataloader=None, test_dataloader=None, voc=None, n_epoch=1000, sample_k=15, device='cuda:0',
                 test_num=5, class_num=20, train_split=2000):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.voc = voc
        self.n_epoch = n_epoch
        self.sample_k = sample_k
        self.device = device
        self.test_num = test_num
        self.clc_num = class_num
        self.train_split = train_split
        self.model.to(self.device)
        self.trainable_paras = []
        self.bert_paras = []
        self.p_w = p_w
        self.tm_name = tm_name
        # self.beta0 = self.init_beta()
        self.beta0 = 0.998
        self.beta = 1.0
        self.contrastive_alpha = contrastive_alpha
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print(k)
                self.trainable_paras.append(v)
        self.optimizer = torch.optim.AdamW(self.trainable_paras, lr=1e-2)
        self.train_num = 0
        #### log file
        log_str = f'contrastive/{tm_name}/{dataname}/{sample_k}/alpha{self.contrastive_alpha}'
        now = int(round(time.time() * 1000))
        now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
        log_str = log_str + '/' + now_time + '/'
        self.log_str = log_str
        if not os.path.exists(log_str):
            os.makedirs(self.log_str)
        self.save_phi = f'{self.log_str}/save_phi'
        if not os.path.exists(self.save_phi):
            os.makedirs(self.save_phi)
        self.save_model_path = f'{self.log_str}/checkpoint'
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.writer = SummaryWriter(self.log_str)

    def init_beta(self):
        """
        Here we consider all data instead of just a batch
        :return:
        """
        print(f'init beta0')
        with torch.no_grad():
            beta_0 = []
            for i, (bow, label, doc, sorted_tfidf, doc_embedding) in enumerate(self.train_dataloader):
                bow = bow.to(self.device)
                sorted_tfidf = sorted_tfidf.to(self.device)
                doc_embedding = doc_embedding.to(self.device)
                bow = bow / (torch.sum(bow, dim=1, keepdim=True) + 1e-10)
                bow_p = bow.clone()
                bow_n = bow.clone()
                rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = self.model.topic_model(
                    bow)
                word_dist = phi_theta_list[0].t()
                theta = theta_list[0]
                ### sample doc
                positive_index = sorted_tfidf[:, -self.sample_k:]
                negetive_index = sorted_tfidf[:, :self.sample_k]
                bow_p = sample_value(bow_p, word_dist, positive_index)
                bow_n = sample_value(bow_n, word_dist, negetive_index)

                # z_p = self.model.topic_model.get_theta(bow_p)
                contextual_z = self.model.encode(doc_embedding)
                contextual_z = F.softmax(contextual_z, dim=1)
                z_n = self.model.topic_model.get_theta(bow_n)
                gamma_l = (theta * contextual_z).sum(1) / (theta * z_n).sum(1)
                beta_0.append(gamma_l.mean())
        beta_0 = torch.mean(torch.stack(beta_0))
        print(f'set beta0 {beta_0}')
        return beta_0

    def train(self):
        train_total_num = self.n_epoch * len(self.train_dataloader)
        best_cluster_purity = 0.
        best_cluster_nmi = 0.
        best_cluster_ami = 0.

        best_mi_f1 = 0.
        best_mi_p = 0.
        best_mi_r = 0.
        best_acc = 0.

        for epoch in range(self.n_epoch):
            self.model.train()
            loss_t = []
            tm_t = []
            c_t = []
            kl_t = []
            like_t = []
            train_theta = None
            train_label = None
            for i, (bow, label, doc, sorted_tfidf, doc_embedding) in enumerate(self.train_dataloader):
                self.train_num += 1
                self.beta = 0.5 - (1.0 / train_total_num) * np.abs(train_total_num / 2.0 - self.train_num) + self.beta0
                bow = bow.to(self.device).float()
                bow_sum = torch.sum(bow, dim=1, keepdim=True)
                sorted_tfidf = sorted_tfidf.to(self.device)
                doc_embedding = doc_embedding.to(self.device).float()
                bow_p = bow.clone()
                bow_n = bow.clone()
                rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = self.model.topic_model(bow)
                if self.tm_name == 'sawtooth':
                    word_dist = phi_theta_list[0].t()
                else:
                    word_dist = phi_theta_list[0] * bow_sum
                theta = theta_list[0]

                ### negative samples
                negetive_index = sorted_tfidf[:, :self.sample_k]
                bow_n = sample_value(bow_n, word_dist, negetive_index)

                #### positive contextual embedding and z
                contextual_z = self.model.encode(doc_embedding)
                positive_z = self.model.topic_model.get_theta_from_embedding(contextual_z)

                #### negative z
                z_n = self.model.topic_model.get_theta(bow_n)

                """
                    make sure z are normalized (for nan case)
                """
                positive_z_norm = positive_z / (torch.sum(positive_z, dim=1, keepdim=True) + 1e-10)
                theta_norm = theta / (torch.sum(theta, dim=1, keepdim=True) + 1e-10)
                z_n_norm = z_n / (torch.sum(z_n, dim=1, keepdim=True) + 1e-10)

                #### loss
                KL = kl_loss_list[0]
                RL = rec_list[0]
                exp_multi_p = torch.exp(theta_norm * positive_z_norm).sum(1)
                exp_multi_n = torch.exp(theta_norm * z_n_norm).sum(1)

                ### contrastive loss
                contrastive_loss = - torch.clamp(torch.log(exp_multi_p / (exp_multi_p + self.beta * exp_multi_n)),
                                                 -1e30, 1e30).mean()
                if torch.isnan(contrastive_loss.sum()):
                    print(f'errorrrrrr')

                #### topic loss
                topic_loss = RL + 1e-1 * KL
                topic_loss = topic_loss.mean()

                loss = 1.0 * topic_loss + self.contrastive_alpha * contrastive_loss
                self.optimizer.zero_grad()
                loss.backward()
                ##### clip grandit
                for para in self.trainable_paras:
                    try:
                        para.grad = para.grad.where(~torch.isnan(para.grad), torch.tensor(0., device=para.grad.device))
                        nn.utils.clip_grad_norm_(para, max_norm=20, norm_type=2)
                    except:
                        pass
                self.optimizer.step()
                loss_t.append(loss.item())
                tm_t.append(topic_loss.item())
                c_t.append(contrastive_loss.item())
                kl_t.append(KL.item())
                like_t.append(RL.item())
                ##### collect theta for downstream task
                if train_theta is None:
                    train_theta = theta.cpu().detach().numpy()
                    train_label = label.cpu().numpy()
                else:
                    train_theta = np.concatenate((train_theta, theta.cpu().detach().numpy()))
                    train_label = np.concatenate((train_label, label.cpu().numpy()))

            self.writer.add_scalar('loss/train', np.mean(loss_t), epoch)
            self.writer.add_scalar('tm_loss/train', np.mean(tm_t), epoch)
            self.writer.add_scalar('con_loss/train', np.mean(c_t), epoch)
            self.writer.add_scalar('kl_loss/train', np.mean(kl_t), epoch)
            self.writer.add_scalar('likelihood/train', np.mean(like_t), epoch)
            if epoch % 10 == 0:
                print(
                    f'epoch {epoch}|{self.n_epoch}: loss: {np.mean(loss_t)}, tm_loss: {np.mean(tm_t)}, contrastive_loss: {np.mean(c_t)}, beta: {self.beta}, kl_loss: {np.mean(kl_t)}, like_loss: {np.mean(like_t)}')

            if epoch % self.test_num == 0:
                self.model.eval()
                loss_t = []
                tm_t = []
                c_t = []
                kl_t = []
                like_t = []
                test_theta = None
                test_label = None
                with torch.no_grad():
                    for i, (bow, label, doc, sorted_tfidf, doc_embedding) in enumerate(self.test_dataloader):

                        bow = bow.to(self.device).float()
                        bow_sum = torch.sum(bow, dim=1, keepdim=True)
                        sorted_tfidf = sorted_tfidf.to(self.device)
                        doc_embedding = doc_embedding.to(self.device).float()
                        bow_p = bow.clone()
                        bow_n = bow.clone()
                        rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = self.model.topic_model(bow)
                        if self.tm_name == 'sawtooth':
                            word_dist = phi_theta_list[0].t()
                        else:
                            word_dist = phi_theta_list[0] * bow_sum
                        theta = theta_list[0]
                        ### sample doc
                        positive_index = sorted_tfidf[:, -self.sample_k:]
                        negetive_index = sorted_tfidf[:, :self.sample_k]
                        bow_n = sample_value(bow_n, word_dist, negetive_index)

                        #### positive contextual embedding and z
                        contextual_z = self.model.encode(doc_embedding)
                        positive_z = self.model.topic_model.get_theta_from_embedding(contextual_z)

                        #### negative z
                        z_n = self.model.topic_model.get_theta(bow_n)

                        """
                            make sure z are normalized (for nan case)
                        """
                        positive_z_norm = positive_z / (torch.sum(positive_z, dim=1, keepdim=True) + 1e-10)
                        theta_norm = theta / (torch.sum(theta, dim=1, keepdim=True) + 1e-10)
                        z_n_norm = z_n / (torch.sum(z_n, dim=1, keepdim=True) + 1e-10)

                        #### loss
                        KL = kl_loss_list[0]
                        RL = rec_list[0]
                        exp_multi_p = torch.exp(theta_norm * positive_z_norm).sum(1)
                        exp_multi_n = torch.exp(theta_norm * z_n_norm).sum(1)
                        contrastive_loss = - torch.clamp(
                            torch.log(exp_multi_p / (exp_multi_p + self.beta * exp_multi_n)), -1e30, 1e30).mean()

                        topic_loss = RL + 1e-1 * KL
                        topic_loss = topic_loss.mean()

                        loss = 1.0 * topic_loss + self.contrastive_alpha * contrastive_loss

                        loss_t.append(loss.item())
                        tm_t.append(topic_loss.item())
                        c_t.append(contrastive_loss.item())
                        kl_t.append(KL.item())
                        like_t.append(RL.item())
                        ##### collect theta for downstream task
                        if test_theta is None:
                            test_theta = theta.cpu().detach().numpy()
                            test_label = label.cpu().numpy()
                        else:
                            test_theta = np.concatenate((test_theta, theta.cpu().detach().numpy()))
                            test_label = np.concatenate((test_label, label.cpu().numpy()))
                #####   clustering
                theta_norm = standardization(test_theta)
                tmp = k_means(theta_norm, self.clc_num)
                predict_label = tmp[1]
                purity_value = purity(test_label, predict_label)
                nmi_value = normalized_mutual_info_score(test_label, predict_label)
                ami_value = adjusted_mutual_info_score(test_label, predict_label)
                if purity_value > best_cluster_purity:
                    best_cluster_purity = purity_value
                    best_cluster_nmi = nmi_value
                    best_cluster_ami = ami_value

                ##### classification
                train_theta_norm = standardization(train_theta)
                clf = LogisticRegression(random_state=0, C=1.0, solver='liblinear', multi_class='ovr', n_jobs=-1).fit(
                    train_theta_norm, train_label)
                pred_label_list = list(clf.predict(theta_norm))
                true_label_list = list(test_label)

                micro_prec, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_label_list,
                                                                                              pred_label_list,
                                                                                              average="micro")
                acc = accuracy_score(true_label_list, pred_label_list)
                if micro_f1_score > best_mi_f1:
                    best_mi_f1 = micro_f1_score
                    best_mi_p = micro_prec
                    best_mi_r = micro_recall
                    best_acc = acc

                print(f'********************test summary **************************')
                print(
                    f'loss: {np.mean(loss_t)}, tm_loss: {np.mean(tm_t)}, contrastive_loss: {np.mean(c_t)}, kl_loss: {np.mean(kl_t)}, like_loss: {np.mean(like_t)}')
                print(
                    f'purity: {purity_value:.4f} / {best_cluster_purity:.4f}, nmi: {nmi_value:.4f}/{best_cluster_nmi:.4f}, ami: {ami_value:.4f}/{best_cluster_ami:.4f}')
                print(
                    f'p: {micro_prec:.4f} / {best_mi_p:.4f}, r: {micro_recall:.4f} / {best_mi_r:.4f}, f1: {micro_f1_score:.4f} / {best_mi_f1:.4f}, acc: {acc:.4f} / {best_acc:.4f}')
                ts_value = ts(phi_list[0], self.p_w)
                print(f'topic specifisity: {ts_value}')
                vision_phi(phi_list, self.voc, self.train_num, outpath=self.save_phi)
                with open(f'{self.save_model_path}/{epoch}_theta_phi_label.pkl', 'wb') as f:
                    pickle.dump([train_theta, train_label, test_theta, test_label, phi_list], f)
                self.writer.add_scalar('loss/test', np.mean(loss_t), epoch)
                self.writer.add_scalar('tm_loss/test', np.mean(tm_t), epoch)
                self.writer.add_scalar('con_loss/test', np.mean(c_t), epoch)
                self.writer.add_scalar('kl_loss/test', np.mean(kl_t), epoch)
                self.writer.add_scalar('likelihood/test', np.mean(like_t), epoch)

                self.writer.add_scalar('purity', purity_value, epoch)
                self.writer.add_scalar('nmi', nmi_value, epoch)
                self.writer.add_scalar('ami', ami_value, epoch)
                self.writer.add_scalar('p', micro_prec, epoch)
                self.writer.add_scalar('r', micro_recall, epoch)
                self.writer.add_scalar('f1', micro_f1_score, epoch)
                self.writer.add_scalar('acc', acc, epoch)
                self.writer.add_scalar('ts', ts_value, epoch)
