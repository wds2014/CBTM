#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:36
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : mydataset.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>

from data_prepare import *


class CustomDataset(Dataset):
    def __init__(self, dataname='20ng', mode='train', voc=None, bert_dim=768, bert_model=None):
        if os.path.exists(f'dataset/{dataname}_dataset/{dataname}.pkl'):
            print(f'load from cached data')
            with open(f'dataset/{dataname}_dataset/{dataname}.pkl', 'rb') as f:
                data = pickle.load(f)
            train_bow = data['train_bow']
            voc = data['voc']
            train_doc = data['train_doc']
            train_label = data['train_label']
            test_bow = data['test_bow']
            test_doc = data['test_doc']
            test_label = data['test_label']
            train_tfidf = data['train_tfidf']
            test_tfidf = data['test_tfidf']
        else:
            print(f'prepare data from {dataname}')
            train_bow, train_label, train_doc, test_bow, test_label, test_doc, voc, train_tfidf, test_tfidf = prepare_data(
                dataname=dataname, voc=voc)

        if os.path.exists(f'dataset/{dataname}_dataset/{dataname}_bert_embedding.pkl'):
            print(f'load from cached bert_embedding data')
            with open(f'dataset/{dataname}_dataset/{dataname}_bert_embedding.pkl', 'rb') as f:
                data = pickle.load(f)
                train_doc_embedding = data['train_doc_embedding']
                test_doc_embedding = data['test_doc_embedding']
        else:
            print(f'create bert embedding from dataset and bert model ***')
            if bert_model is None:
                print(f'bert model need !')
                exit()
            else:
                train_doc_embedding = np.zeros([0, bert_dim])
                ### train
                batch = 200
                train_doc_num = len(train_doc)
                batch_num = train_doc_num // batch
                for i in range(batch_num):
                    out = bert_model.encode(train_doc[i * batch: (i + 1) * batch])
                    train_doc_embedding = np.concatenate((train_doc_embedding, np.array(out)))
                #### last:
                out = bert_model.encode(train_doc[(i + 1) * batch:])
                try:
                    train_doc_embedding = np.concatenate((train_doc_embedding, np.array(out)))
                except:
                    train_doc_embedding = train_doc_embedding
                assert len(train_doc_embedding) == len(train_doc)

                #### test
                test_doc_embedding = np.zeros([0, bert_dim])
                test_doc_num = len(test_doc)
                batch_num = test_doc_num // batch
                for i in range(batch_num):
                    out = bert_model.encode(test_doc[i * batch: (i + 1) * batch])
                    test_doc_embedding = np.concatenate((test_doc_embedding, np.array(out)))

                out = bert_model.encode(test_doc[(i + 1) * batch:])
                try:
                    test_doc_embedding = np.concatenate((test_doc_embedding, np.array(out)))
                except:
                    test_doc_embedding = test_doc_embedding
                assert len(test_doc_embedding) == len(test_doc)
                with open(f'dataset/{dataname}_dataset/{dataname}_bert_embedding.pkl', 'wb') as f:
                    pickle.dump({'train_doc_embedding': train_doc_embedding, 'test_doc_embedding': test_doc_embedding},
                                f)
                # self.train_doc_embedding = train_doc_embedding
                # self.test_doc_embedding = test_doc_embedding

        if mode == 'train':
            self.bow = train_bow
            self.doc = train_doc
            self.doc_embedding = train_doc_embedding
            self.label = train_label
            self.sorted_tfidf = np.argsort(-train_tfidf.toarray(), axis=1)
        elif mode == 'test':
            self.bow = test_bow
            self.doc = test_doc
            self.doc_embedding = test_doc_embedding
            self.label = test_label
            self.sorted_tfidf = np.argsort(-test_tfidf.toarray(), axis=1)
        else:
            train_bow = train_bow.toarray()
            test_bow = test_bow.toarray()
            bow = np.concatenate((train_bow, test_bow), axis=0)
            train_tfidf = train_tfidf.toarray()
            test_tfidf = test_tfidf.toarray()
            tfidf = np.concatenate((train_tfidf, test_tfidf), axis=0)
            self.bow = sparse.csc_matrix(bow)
            self.label = train_label + test_label
            self.doc = train_doc + test_doc
            self.doc_embedding = np.concatenate((train_doc_embedding, test_doc_embedding))
            self.sorted_tfidf = np.argsort(-tfidf, axis=1)
        self.voc = voc
        assert self.bow.shape[0] == len(self.label)
        assert len(self.doc) == len(self.label)

        self.train_num = train_bow.shape[0]
        self.test_num = test_bow.shape[0]
        try:
            pw = train_bow.toarray()
        except:
            pw = train_bow
        self.pw = np.sum(pw, axis=0) / np.sum(pw)

        self.N, self.vocab_size = self.bow.shape

    def __getitem__(self, index):
        return np.squeeze(self.bow[index].toarray()), np.squeeze(np.array(self.label[index])), self.doc[
            index], np.squeeze(self.sorted_tfidf[index]), np.squeeze(self.doc_embedding[index])

    def __len__(self):
        return self.N


def data_loader(bert_model=None, dataname='20ng', voc=None, mode='train', batch_size=200, shuffle=True, drop_out=True,
                num_workers=4):
    dataset = CustomDataset(dataname=dataname, mode=mode, voc=voc, bert_model=bert_model)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_out), dataset.vocab_size, dataset.voc, dataset.train_num, dataset.pw
