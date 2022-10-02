#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:41
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : ntms.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>


import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class SAWtooth_layer(nn.Module):
    def __init__(self, alpha, rho, d_in=768, d_dim=256, pre_topic_k=100, is_top=False, is_bottom=False, learn_prior=True):
        super(SAWtooth_layer, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(100.0).float()
        self.wei_shape_min = torch.tensor(0.1).float()
        self.theta_max = torch.tensor(1000.0).float()
        self.is_top = is_top
        self.is_bottom = is_bottom

        self.alpha = alpha   ### v,d
        self.rho = rho         #### k,d
        self.topic_k = self.rho.shape[0]
        self.h_encoder = nn.Linear(d_in, d_dim)    ### h_hiddent encoder
        self.bn_layer = nn.BatchNorm1d(d_dim)
        if is_top:
            self.shape_scale_encoder = nn.Linear(d_dim, 2*self.topic_k)
            self.k_prior = torch.ones((1, self.topic_k))
            self.l_prior = 0.1 * torch.ones((1, self.topic_k))
            if torch.cuda.is_available():
                self.k_prior = self.k_prior.cuda()
                self.l_prior = self.l_prior.cuda()
            if learn_prior:
                self.k_prior = nn.Parameter(self.k_prior)
                self.l_prior = nn.Parameter(self.l_prior)
        else:
            self.shape_scale_encoder = nn.Linear(d_dim+pre_topic_k, 2*self.topic_k)     ### for h + phi_{t+1}*theta_{t+1}

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.cuda()))

    def reparameterize(self, Wei_shape, Wei_scale, sample_num=50):
        eps = torch.cuda.FloatTensor(sample_num, Wei_shape.shape[0], Wei_shape.shape[1]).uniform_(0.0, 1.)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(sample_num, 1, 1) * torch.pow(-self.log_max(1 - eps), \
                            torch.unsqueeze(1 / Wei_shape, axis=0).repeat(sample_num, 1, 1))
        return torch.mean(torch.clamp(theta, self.real_min.cuda(), self.theta_max), dim=0, keepdim=False)

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.cuda() * Gam_shape * 1 / Wei_shape + self.log_max(
            1 / Wei_shape)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
        part3 = eulergamma.cuda() + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1.))
        # if torch.isinf(likelihood):
        #     x = 1.0 * x
        #     x[torch.where(x < 0)] == torch.tensor(120.0).cuda()
        #     likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1.))
        #     print(likelihood.item())
        return -likelihood / x.shape[1]

    def get_phi(self):
        w = torch.mm(self.alpha, self.rho.t())   ### v,k
        return torch.softmax(w+self.real_min.cuda(), dim=0)
    def decoder(self, x):
        phi = self.get_phi()
        return torch.mm(phi, x), phi

    def forward(self, x, prior=None, bow=None):
        #### x: the special token 'TOPIC' hidden embedding,  batch, h_dim
        #### prior: phi_{t+1} * theta_{t+1}: k, batch
        #### bow: raw txt bow form n,v
        rec_loss = None
        kl_loss = None
        hidden = F.relu(self.bn_layer(self.h_encoder(x)))
        if not self.is_top:
            hidden = torch.cat((hidden, prior.t()), 1)     ### batch, (d_dim+pre_topic_k)

        k_temp, l_temp = torch.chunk(self.shape_scale_encoder(hidden), 2, dim=1)
        k = torch.clamp(F.softplus(k_temp), self.wei_shape_min, self.wei_shape_max)
        l_temp = F.softplus(l_temp) / torch.exp(torch.lgamma(1 + 1 / k))
        l = torch.clamp(l_temp, self.real_min, 9999.0)

        theta = self.reparameterize(k, l)       ### n,k
        phi_theta, phi = self.decoder(theta.t())   #### v,n
        if self.is_top:
            kl_loss = self.KL_GamWei(F.softplus(self.k_prior).t(),
                                     F.softplus(self.l_prior).t(),k.t(), l.t())
        else:
            kl_loss = self.KL_GamWei(prior, torch.tensor(1.0, dtype=torch.float32).cuda(),
                           k.t(), l.t())
        if self.is_bottom:
            if bow is not None:
                rec_loss = self.compute_loss(bow.t(), phi_theta)
            else:
                rec_loss = 0.
        return rec_loss, kl_loss, phi.cpu().detach().numpy(), phi_theta, theta


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).cuda()
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).cuda())
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class ProdLDA_infer(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, act='softplus', dropout=0.2):
        """
        ProdLDA
        input_size: batch, h (768), hidden embedding of TOPIC token
        output_size: k
        hidden_size: tuple, length = n_layer
        """
        super(ProdLDA_infer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if act == "softplus":
            self.activation = nn.Softplus()
        else:
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward_from_embedding(self, x):
        return self.f_mu_batchnorm(self.f_mu(x)), self.f_sigma_batchnorm(self.f_sigma(x))

    def forward(self, x):
        #### x: batch, h
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

class ProdLDA(nn.Module):
    """
    Autoencoding Variational Inference for Topic Models, in ICLR 2017
    """

    def __init__(self, num_topic=100, voc_size=2000, hidden_sizes=(100, 100), dropout=0.2, learn_priors=True, device='cuda:0'):
        super(ProdLDA, self).__init__()
        # encoder
        self.voc_size = voc_size
        self.num_topic = num_topic
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.device = device

        self.infnet = ProdLDA_infer(self.voc_size, self.num_topic, hidden_sizes, act='softplus', dropout=dropout)
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * self.num_topic)
        self.prior_mean = self.prior_mean.to(self.device)
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.num_topic)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * self.num_topic)
        self.prior_variance = self.prior_variance.to(self.device)
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(self.num_topic, self.voc_size)
        self.beta = self.beta.to(self.device)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(self.voc_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    def reparameterize(self, mu, logvar, sample_num=20):
        """Reparameterize the theta distribution."""
        logvar_sample = logvar.unsqueeze(0).repeat([sample_num, 1, 1])
        mu_sample = mu.unsqueeze(0).repeat([sample_num, 1, 1])
        std = torch.exp(0.5 * logvar_sample)
        eps = torch.randn_like(std)
        return (eps.mul(std).add_(mu_sample)).mean(0)


    def forward(self, x):
        posterior_mu, posterior_log_sigma = self.infnet(x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        # theta = self.drop_theta(theta)
        word_dist = torch.matmul(theta, self.beta)
        word_dist_loss = F.softmax(
            self.beta_batchnorm(word_dist), dim=1)
        KL, RL = self.loss(x, word_dist_loss, self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma)
        return [RL], [KL], [F.softmax(self.beta.t(), dim=0).cpu().detach().numpy()], [word_dist_loss], [theta]


    def loss(self, bow, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        prior_variance = F.softplus(prior_variance)
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = torch.sum(0.5 * (
                var_division + diff_term - self.num_topic + logvar_det_division), dim = -1).mean()

        # Reconstruction term
        RL = -torch.sum(bow * torch.log(word_dists + 1e-10), dim=-1).mean()

        # loss = self.weights["beta"]*KL + RL

        return KL, RL

    def get_theta_from_embedding(self, embedding):
        posterior_mu, posterior_log_sigma = self.infnet.forward_from_embedding(embedding)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

        return theta

    def get_theta(self, x_norm):
        posterior_mu, posterior_log_sigma = self.infnet(x_norm)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

        return theta

class ETM(nn.Module):
    """
    topic model in word embedding space
    """

    def __init__(self, num_topics=100, vocab_size=2000, t_hidden_size=100, rho_size=100, emsize=100,
                 theta_act='softplus', embeddings=None, train_embeddings=True, enc_drop=0.5, device='cuda:0'):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)
        self.f_mu_batchnorm = nn.BatchNorm1d(num_topics, affine=False)
        self.f_sigma_batchnorm = nn.BatchNorm1d(num_topics, affine=False)

        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        print(vocab_size, " THE Vocabulary size is here ")
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Sequential(
            nn.Linear(t_hidden_size, num_topics, bias=True),
            self.f_mu_batchnorm)
        self.logsigma_q_theta = nn.Sequential(
            nn.Linear(t_hidden_size, num_topics, bias=True),
            self.f_sigma_batchnorm)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar, sample_num=20):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            logvar_sample = logvar.unsqueeze(0).repeat([sample_num, 1, 1])
            mu_sample = mu.unsqueeze(0).repeat([sample_num, 1, 1])
            std = torch.exp(0.5 * logvar_sample)
            eps = torch.randn_like(std)
            return (eps.mul_(std).add_(mu_sample)).mean(0)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.
        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        z = torch.clamp(z, 1e-20, 1e1)
        theta = F.softmax(z, dim=-1)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return theta, kl_theta

    def get_beta(self):
        """
        This generate the description as a defintion over words
        Returns:
            [type]: [description]
        """
        try:
            logit = self.alphas(self.rho.weight)  # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).t()  ## softmax over vocab dimension
        return beta

    def get_theta_from_embedding(self, embedding):
        mu_theta = self.mu_q_theta(embedding)
        logsigma_theta = self.logsigma_q_theta(embedding)
        z = self.reparameterize(mu_theta, logsigma_theta)
        z = torch.clamp(z, 1e-20, 1e1)
        theta = F.softmax(z, dim=-1)
        return theta

    def get_theta(self, bows):
        """
        getting the topic poportion for the document passed in the normalixe bow or tf-idf"""
        normalized_bows = bows / (torch.sum(bows, dim=1, keepdim=True) + 1e-10)
        # normalized_bows = bows
        theta, kld_theta = self.encode(normalized_bows)

        return theta

    def decode(self, theta, beta):
        """compute the probability of topic given the document which is equal to theta^T ** B
        Args:
            theta ([type]): [description]
            beta ([type]): [description]
        Returns:
            [type]: [description]
        """
        res = torch.mm(theta, beta)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        predictions = torch.log(results_without_zeros)
        return predictions

    def forward(self, bows, theta=None, aggregate=True):
        ## get \theta
        normalized_bows = bows / (torch.sum(bows, dim=1, keepdim=True) + 1e-10)
        # normalized_bows = bows
        theta, kld_theta = self.encode(normalized_bows)
        # if theta is None:
        #     theta, kld_theta = self.get_theta(normalized_bows)
        # else:
        #     kld_theta = None

        ## get \beta
        beta = self.get_beta()   ## k,v

        ## get prediction loss
        # preds = self.decode(theta, beta)
        preds = torch.mm(theta, beta)
        recon_loss = -(torch.log(preds+1e-10) * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return [recon_loss],[kld_theta], [beta.t().cpu().detach().numpy()], [preds], [theta]


class Sawtooth(nn.Module):
    ## doneeeeeee
    def __init__(self, k=[256,128,64], h=[256,128,64], v=2000, emb_dim=100, add_embedding=True):
        super(Sawtooth, self).__init__()
        self.voc_size = v
        self.hidden_size = h
        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in self.hidden_size])
        self.layer_num = len(k)
        self.dropout = nn.Dropout(p=0.1)
        self.add_embedding = add_embedding
        if add_embedding:
            self.embed_layer = Conv1D(self.hidden_size[0], 1, self.voc_size)

        h_encoder = [Conv1D(self.hidden_size[0], 1, self.hidden_size[0])]
        for i in range(self.layer_num - 1):
            h_encoder.append(Conv1D(self.hidden_size[i + 1], 1, self.hidden_size[i]))
        self.h_encoder = nn.ModuleList(h_encoder)
        self.alpha = self.init_alpha([v] + k, emb_dim)
        self.layer_num = len(k)
        sawtooth_layer = []
        for idx, each_k in enumerate(k):
            if idx == 0:
                if len(k) == 1:
                    sawtooth_layer.append(SAWtooth_layer(self.alpha[0], self.alpha[1], d_in=h[0],
                                                         d_dim=h[0], pre_topic_k=k[idx],
                                                         is_top=True, is_bottom=True))
                else:
                    sawtooth_layer.append(SAWtooth_layer(self.alpha[0], self.alpha[1], d_in=h[0],
                                                         d_dim=h[0], pre_topic_k=k[idx],
                                                         is_top=False, is_bottom=True))
            elif idx == len(k) - 1:
                sawtooth_layer.append(SAWtooth_layer(self.alpha[-2], self.alpha[-1], d_in=h[idx],
                                                     d_dim=h[idx], pre_topic_k=None,
                                                     is_top=True, is_bottom=False))
            else:
                sawtooth_layer.append(SAWtooth_layer(self.alpha[idx], self.alpha[idx + 1], d_in=h[idx],
                                                     d_dim=h[idx], pre_topic_k=k[idx],
                                                     is_top=False, is_bottom=False))
        self.sawtooth_layer = nn.ModuleList(sawtooth_layer)

    def init_alpha(self, k, emb_dim):
        w_para = []
        for idx, each_topic in enumerate(k):
            w = torch.ones(each_topic, emb_dim).cuda()
            nn.init.normal_(w, std=0.02)
            # w_para.append(torch.Tensor.requires_grad_(w))
            w_para.append(Parameter(w))
        return w_para

    def res_block(self, x, layer_num):
        ### res block for hidden path
        x1 = self.h_encoder[layer_num](x)
        try:
            out = x + x1
        except:
            out = x1
        return self.dropout(F.relu(self.bn_layer[layer_num](out)))

    def forward(self, x):
        hidden_list = [0] * self.layer_num
        theta_list = [0] * self.layer_num
        phi_list = [0] * self.layer_num
        phi_theta_list = [0] * (self.layer_num+1)
        kl_loss_list = [0] * self.layer_num

        rec_list = [0] * self.layer_num
        #### upward path
        if self.add_embedding:
            x_embed = self.embed_layer(1.0*x)
        else:
            x_embed = 1.0*x
        for t in range(self.layer_num):
            if t == 0:
                hidden_list[t] = self.res_block(x_embed, t)
            else:
                hidden_list[t] = self.res_block(hidden_list[t-1], t)
        #### downward path
        for t in range(self.layer_num-1, -1, -1):
            rec_list[t], kl_loss_list[t], phi_list[t], phi_theta_list[t], \
            theta_list[t] = self.sawtooth_layer[t](hidden_list[t], phi_theta_list[t+1], x)

        return rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list

    def get_theta_from_embedding(self, embedding, x=None):
        hidden_list = [0] * self.layer_num
        theta_list = [0] * self.layer_num
        phi_list = [0] * self.layer_num
        phi_theta_list = [0] * (self.layer_num+1)
        kl_loss_list = [0] * self.layer_num

        rec_list = [0] * self.layer_num
        #### upward path
        for t in range(self.layer_num):
            if t == 0:
                hidden_list[t] = embedding
            else:
                hidden_list[t] = embedding
        #### downward path
        for t in range(self.layer_num-1, -1, -1):
            rec_list[t], kl_loss_list[t], phi_list[t], phi_theta_list[t], \
            theta_list[t] = self.sawtooth_layer[t](hidden_list[t], phi_theta_list[t+1], x)

        return theta_list[0]


    def get_theta(self, x):
        hidden_list = [0] * self.layer_num
        theta_list = [0] * self.layer_num
        phi_list = [0] * self.layer_num
        phi_theta_list = [0] * (self.layer_num+1)
        kl_loss_list = [0] * self.layer_num

        rec_list = [0] * self.layer_num
        #### upward path
        if self.add_embedding:
            x_embed = self.embed_layer(1.0*x)
        else:
            x_embed = 1.0*x
        for t in range(self.layer_num):
            if t == 0:
                hidden_list[t] = self.res_block(x_embed, t)
            else:
                hidden_list[t] = self.res_block(hidden_list[t-1], t)
        #### downward path
        for t in range(self.layer_num-1, -1, -1):
            rec_list[t], kl_loss_list[t], phi_list[t], phi_theta_list[t], \
            theta_list[t] = self.sawtooth_layer[t](hidden_list[t], phi_theta_list[t+1], x)

        return theta_list[0]


class read_20ng(Dataset):
    def __init__(self, path = '/home/wds/2021/ACT/dataset/20News/20ng_glove.pkl'):
        with open(path,'rb') as f:
            data = pickle.load(f)
        self.train_data = data['bow']
        self.voc = data['voc']
        self.N, self.voc_size = self.train_data.shape

    def __getitem__(self, index):
        try:
            return np.squeeze(self.train_data[index].toarray())
        except:
            return np.squeeze(self.train_data[index])

    def __len__(self):
        return self.N
def get_train_loader_tm(batch_size=200, device='cuda:0', shuffle=True, num_workers=2):
    dataset = read_20ng()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=False), dataset.voc_size, dataset.voc
def vision_phi(Phi, voc, train_num, outpath='phi_output', top_n=30):
    def get_top_n(phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += voc[index]
            top_n_words += ' '
        return top_n_words

    outpath = outpath + '/' + str(train_num)
    # Phi = Phi[::-1]
    if voc is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        phi = 1
        for num, phi_layer in enumerate(Phi):
            phi = np.dot(phi, phi_layer)
            phi_k = phi.shape[1]
            path = os.path.join(outpath, 'phi' + str(num) + '.txt')
            f = open(path, 'w')
            for each in range(phi_k):
                top_n_words = get_top_n(phi[:, each], top_n)
                f.write(top_n_words)
                f.write('\n')
            f.close()
    else:
        print('voc need !!')

if __name__ == "__main__":
    dataloader, voc_size, voc = get_train_loader_tm()
    k = [100]
    h = [100]
    model = Sawtooth(k=k, h=h, v=voc_size, emb_dim=100, add_embedding=True)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    kl_weight = [1.0 for _ in range(len(k))]
    n_epoch = 12000
    for epoch in range(n_epoch):
        loss_t = []
        kl_loss_t = []
        rl_loss_t = []
        for i, data in enumerate(dataloader):
            data = data.cuda().float()
            # data = data / torch.sum(data,dim=1, keepdim=True)
            rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(data)
            rec_loss = rec_list[0]
            kl_part = torch.sum(torch.tensor([weight * kl_loss for weight, kl_loss in zip(kl_weight, kl_loss_list)]))
            loss = 1e-2 * kl_part + rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_t.append(loss.item())
            kl_loss_t.append(kl_part.item())
            rl_loss_t.append(rec_loss.item())
        print(f'epoch{epoch}|{n_epoch} loss: {np.mean(loss_t)}, kl_loss: {np.mean(kl_loss_t)}, rl_loss: {np.mean(rl_loss_t)}')
        vision_phi(phi_list, voc, epoch, outpath=f'phi_output_sawtooth/phi_output_{epoch}',top_n=30)