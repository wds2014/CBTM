#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:36
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : data_prepare.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>


import numpy as np
import re
import string
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle



def vision_phi(Phi, voc, train_num, outpath='phi_output', top_n=50, topic_diversity=True):
    def get_diversity(topics):
        word = []
        for line in topics:
            word += line
        word_unique = np.unique(word)
        return len(word_unique) / len(word)
    if voc is not None:
        outpath = outpath + '/' + str(train_num)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        phi = 1
        for num, phi_layer in enumerate(Phi):
            phi = np.dot(phi, phi_layer)
            phi_k = phi.shape[1]
            path = os.path.join(outpath, 'phi' + str(num) + '.txt')
            f = open(path, 'w')
            topic_word = []
            for each in range(phi_k):
                top_n_words = get_top_n(phi[:, each], top_n, voc)
                topic_word.append(top_n_words.split()[:25])
                f.write(top_n_words)
                f.write('\n')
            f.close()
            if topic_diversity:
                td_value = get_diversity(topic_word)
            print('topic diversity at layer {}: {}'.format(num, td_value))
    else:
        print('voc need !!')


def get_top_n(phi, top_n, voc):
    top_n_words = ''
    idx = np.argsort(-phi)
    for i in range(top_n):
        index = idx[i]
        top_n_words += voc[index]
        top_n_words += ' '
    return top_n_words


class Tokenizer(object):
    """
    Text tokenization methods
    """

    # Default punctuation list
    PUNCTUATION = string.punctuation

    # English Stop Word List (Standard stop words used by Apache Lucene)
    STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
                  "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
                  "they", "this", "to", "was", "will", "with"}

    @staticmethod
    def tokenize(text):
        """
        Tokenizes input text into a list of tokens. Filters tokens that match a specific pattern and removes stop words.
        Args:
            text: input text
        Returns:
            list of tokens
        """

        # Convert to all lowercase, split on whitespace, strip punctuation
        tokens = [token.strip(Tokenizer.PUNCTUATION) for token in text.lower().split()]

        # Tokenize on alphanumeric strings.
        # Require strings to be at least 2 characters long.
        # Require at least 1 alpha character in string.
        return [token for token in tokens if re.match(r"^\d*[a-z][\-.0-9:_a-z]{1,}$", token) and token not in Tokenizer.STOP_WORDS]

def prepare_data(dataname='20ng', voc=None, save=True, voc_size=None):
    """
    txt to bow, and tfidf
    each line is a document
    :param dataname: 20ng.txt
    :param voc: list or dic
    :param save: True for saving resulting,
    :return: sparse bow, sparse tfidf, list label, and sequence doc
    """
    if dataname == '20ng':
        label_name = ["rec.motorcycles", "talk.politics.guns", "talk.politics.misc", "soc.religion.christian",
                      "sci.electronics", "comp.graphics", "talk.religion.misc", "comp.windows.x",
                      "comp.sys.ibm.pc.hardware", "alt.atheism", "sci.med", "sci.crypt",
                      "sci.space", "misc.forsale", "rec.sport.hockey", "rec.sport.baseball",
                      "talk.politics.mideast", "comp.os.ms-windows.misc", "rec.autos", "comp.sys.mac.hardware"]
        label_1 = ['comp.graphics', 'comp.windows.x', 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc',
                   'comp.sys.mac.hardware']
        label_2 = ['misc.forsale']
        label_3 = ['rec.motorcycles', 'rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
        label_4 = ['talk.politics.guns', 'talk.politics.misc', 'talk.politics.mideast']
        label_5 = ['sci.electronics', 'sci.med', 'sci.crypt', 'sci.space']
        label_6 = ['soc.religion.christian', 'talk.religion.misc', 'alt.atheism']
        # voc_size=2000
    elif dataname == '20ng_6':
        label_name = ["rec.motorcycles", "talk.politics.guns", "talk.politics.misc", "soc.religion.christian",
                      "sci.electronics", "comp.graphics", "talk.religion.misc", "comp.windows.x",
                      "comp.sys.ibm.pc.hardware", "alt.atheism", "sci.med", "sci.crypt",
                      "sci.space", "misc.forsale", "rec.sport.hockey", "rec.sport.baseball",
                      "talk.politics.mideast", "comp.os.ms-windows.misc", "rec.autos", "comp.sys.mac.hardware"]
        label_1 = ['comp.graphics', 'comp.windows.x', 'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc',
                   'comp.sys.mac.hardware']
        label_2 = ['misc.forsale']
        label_3 = ['rec.motorcycles', 'rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
        label_4 = ['talk.politics.guns', 'talk.politics.misc', 'talk.politics.mideast']
        label_5 = ['sci.electronics', 'sci.med', 'sci.crypt', 'sci.space']
        label_6 = ['soc.religion.christian', 'talk.religion.misc', 'alt.atheism']
    elif dataname == "r8":
        label_name = ['money-fx', 'trade', 'acq', 'ship', 'crude', 'interest', 'earn', 'grain']
        voc_size = 7688
    elif dataname == "r52":
        label_name = ['jobs', 'bop', 'copper', 'meal-feed', 'strategic-metal',
                    'trade', 'grain', 'lumber', 'orange', 'gnp', 'heat', 'sugar', 'tin', 'pet-chem',
                    'money-fx', 'tea', 'gas', 'earn', 'cpu', 'ship', 'coffee', 'housing', 'cocoa', 'jet',
                    'platinum', 'iron-steel', 'instal-debt', 'cotton', 'acq', 'gold', 'lei', 'dlr', 'zinc',
                    'potato', 'carcass', 'fuel', 'ipi', 'crude', 'alum', 'nat-gas', 'retail', 'rubber',
                    'nickel', 'wpi', 'interest', 'livestock', 'cpi', 'reserves', 'money-supply', 'veg-oil',
                    'income', 'lead']
        voc_size = 8892
    elif dataname == "mr":
        label_name = ['0', '1']
        voc_size = 18764
    elif dataname == "ohsumed":
        label_name = ['C02', 'C20', 'C23', 'C03',
                'C09', 'C01', 'C22', 'C13', 'C11', 'C04', 'C12', 'C19', 'C07', 'C15',
                'C14', 'C17', 'C08', 'C10', 'C21', 'C16', 'C05', 'C06', 'C18']
        voc_size = 14157
    elif dataname == 'agnews':
        label_name = ['1', '2', '3', '4']
        # voc_size = 20000
    elif dataname == 'wiki103':
        label_name = ['wdsss']
        # voc_size = 2000
    elif dataname == 'Biomedical':
        label_name = [str(i) for i in range(1,21)]
        voc_size = 20000
    elif dataname == 'GoogleNews':
        label_name = [str(i) for i in range(1,153)]
        voc_size = 20000
    elif dataname == 'StackOverflow':
        label_name = [str(i) for i in range(1,21)]
        voc_size = 20000
    elif dataname == 'dpsubset':
        label_name = [str(i) for i in range(14)]
        # voc_size = 20000

    train_doc_path = f'dataset/{dataname}_dataset/{dataname}_train_clean.txt'
    train_label_path = f'dataset/{dataname}_dataset/{dataname}_train_clean_label.txt'
    test_doc_path = f'dataset/{dataname}_dataset/{dataname}_test_clean.txt'
    test_label_path = f'dataset/{dataname}_dataset/{dataname}_test_clean_label.txt'
    print(f'read txt data from {dataname}')
    with open(train_doc_path, encoding="utf-8") as f:
        train_doc = f.readlines()
    with open(train_label_path, encoding="utf-8") as f:
        train_label = f.readlines()
    with open(test_doc_path, encoding="utf-8") as f:
        test_doc = f.readlines()
    with open(test_label_path, encoding="utf-8") as f:
        test_label = f.readlines()
    print(f'load data doneeee, train_doc: {len(train_doc)}, test_doc: {len(test_doc)}')


    train_doc_list = []
    train_label_list = []
    test_doc_list = []
    test_label_list = []
    if dataname == '20ng':
        for id_, (text, label) in enumerate(zip(train_doc, train_label)):
            text = text.strip()
            _, mode, label_ = label.strip().split('\t')
            label_index = label_name.index(label_)
            train_doc_list.append(text)
            train_label_list.append(label_index)

        for id_, (text, label) in enumerate(zip(test_doc, test_label)):
            text = text.strip()
            _, mode, label_ = label.strip().split('\t')
            label_index = label_name.index(label_)
            test_doc_list.append(text)
            test_label_list.append(label_index)
    elif dataname == '20ng_6':
        for id_, (text, label) in enumerate(zip(train_doc, train_label)):
            text = text.strip()
            _, mode, label_ = label.strip().split('\t')

            if label_ in label_1:
                train_label_list.append(0)
            elif label_ in label_2:
                train_label_list.append(1)
            elif label_ in label_3:
                train_label_list.append(2)
            elif label_ in label_4:
                train_label_list.append(3)
            elif label_ in label_5:
                train_label_list.append(4)
            elif label_ in label_6:
                train_label_list.append(5)
            train_doc_list.append(text)

        for id_, (text, label) in enumerate(zip(test_doc, test_label)):
            text = text.strip()
            _, mode, label_ = label.strip().split('\t')
            test_doc_list.append(text)

            if label_ in label_1:
                test_label_list.append(0)
            elif label_ in label_2:
                test_label_list.append(1)
            elif label_ in label_3:
                test_label_list.append(2)
            elif label_ in label_4:
                test_label_list.append(3)
            elif label_ in label_5:
                test_label_list.append(4)
            elif label_ in label_6:
                test_label_list.append(5)
    else:
        for id_, (text, label) in enumerate(zip(train_doc, train_label)):
            text = text.strip()
            label_ = label.strip().split()[0]
            label_index = label_name.index(label_)
            train_doc_list.append(text)
            train_label_list.append(label_index)

        for id_, (text, label) in enumerate(zip(test_doc, test_label)):
            text = text.strip()
            label_ = label.strip().split()[0]
            label_index = label_name.index(label_)
            test_doc_list.append(text)
            test_label_list.append(label_index)

    if voc is not None:
        if isinstance(voc,list):
            voc_dict = {word: idx for idx, word in enumerate(voc)}
        elif isinstance(voc, dict):
            #### todo makes sure voc is sorted
            voc_dict = voc
            voc = voc.keys()
        else:
            print('unknown voc type')
            exit()
        vectorizer = CountVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        tfidfer = TfidfVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        train_bow = vectorizer.fit_transform(train_doc_list)
        test_bow = vectorizer.fit_transform(test_doc_list)
        train_tfidf = tfidfer.fit_transform(train_doc_list)
        test_tfidf = tfidfer.fit_transform(test_doc_list)
    elif voc_size:
        print(f'build voc from {dataname}, just use top-{voc_size} words')
        vectorizer = CountVectorizer(max_features=voc_size, tokenizer=Tokenizer.tokenize)
        train_bow = vectorizer.fit_transform(train_doc_list+test_doc_list)
        voc_dict = vectorizer.vocabulary_
        voc = sorted(voc_dict.items(), key=lambda x: x[1], reverse=False)
        voc = [each[0] for each in voc]
        vectorizer = CountVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        train_bow = vectorizer.fit_transform(train_doc_list)
        test_bow = vectorizer.fit_transform(test_doc_list)
        tfidfer = TfidfVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        train_tfidf = tfidfer.fit_transform(train_doc_list)
        test_tfidf = tfidfer.fit_transform(test_doc_list)
    else:
        print(f'no define voc and voc_size for {dataname}, just use max_df: 0.9, and min_df: 20')
        vectorizer = CountVectorizer(max_df=0.9, min_df=20, tokenizer=Tokenizer.tokenize)
        train_bow = vectorizer.fit_transform(train_doc_list + test_doc_list)
        voc_dict = vectorizer.vocabulary_
        voc = sorted(voc_dict.items(), key=lambda x: x[1], reverse=False)
        voc = [each[0] for each in voc]
        vectorizer = CountVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        train_bow = vectorizer.fit_transform(train_doc_list)
        test_bow = vectorizer.fit_transform(test_doc_list)
        tfidfer = TfidfVectorizer(vocabulary=voc_dict, tokenizer=Tokenizer.tokenize)
        train_tfidf = tfidfer.fit_transform(train_doc_list)
        test_tfidf = tfidfer.fit_transform(test_doc_list)
        print(f'the voc size is : {len(voc)}')

    if save:
        with open(f'dataset/{dataname}_dataset/{dataname}.pkl', 'wb') as f:
            pickle.dump({'train_bow': train_bow, 'train_label': train_label_list, 'train_doc': train_doc_list,
                         'test_bow': test_bow, 'test_label': test_label_list, 'test_doc': test_doc_list, 'voc': voc,
                         'train_tfidf': train_tfidf, 'test_tfidf': test_tfidf}, f)
    return train_bow, train_label_list, train_doc_list, test_bow, test_label_list, test_doc_list, voc, train_tfidf, test_tfidf
