from random import choice
import json
import tensorflow as tf
from tqdm import tqdm
import numpy as np

max_len = 128

word2id = open('./data_trans/word2id.txt', 'r', encoding='utf-8')
id2predicate, predicate2id = json.load(open('./data_trans/all_50_schemas_me.json', encoding='utf-8'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
num_classes = len(id2predicate)
word_list = [key.strip('\n') for key in word2id]

def Token(text):
    text2id = []
    for word in text:
        if word in word_list:
            text2id.append(word_list.index(word))
        else:
            word = '[UNK]'
            text2id.append(word_list.index(word))
    return text2id


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1
'''
ner:预测subject
perdicate:预测object和relation矩阵(128*num_class)
'''
def get_input(data):
    input_x, input_ner, input_re1, input_re2, seq_len, position_mask = [], [], [], [], [], []
    for l in tqdm(range(64000)):
        items = {}
        line = data[l]
        text = line['text'][:128]
        spo = line['spo_list']
        text2id = Token(text)
        for sp in spo:
            sp = (Token(sp[0]), sp[1], Token(sp[2]))
            subjectid = list_find(text2id, sp[0])
            objectid = list_find(text2id, sp[2])
            if subjectid != -1 and objectid != -1:
                key = (subjectid, subjectid + len(sp[0]))
                if key not in items:
                    items[key] = []
                items[key].append((objectid,
                                   objectid + len(sp[2]),
                                   predicate2id[sp[1]]))
        if items:
            input_x.append(text2id)
            #seq_len.append(len(text2id))
            ner_s = np.zeros(len(text2id))
            p_mask = np.zeros(len(text2id))
            for j in items:
                ner_s[j[0]] = 1
                ner_s[j[0]+1:j[1]] = 2
            #print(ner_s)
            input_ner.append(ner_s)
            k1, k2 = np.array(list(items.keys())).T
            k1 = choice(k1)
            k2 = choice(k2[k2 >= k1])
            er_s1 = np.zeros((128, num_classes), dtype=np.float32)
            er_s2 = np.zeros((128, num_classes), dtype=np.float32)
            p_mask[k1:k2] = 1
            position_mask.append(p_mask)
            for j in items.get((k1, k2), []):
                er_s1[j[0]][j[2]] = 1
                er_s2[j[1] - 1][j[2]] = 1
            input_re1.append(er_s1)
            input_re2.append(er_s2)

    #seq_len = np.array(seq_len, dtype=np.int32)
    input_re1 = np.array(input_re1, dtype=np.float32)
    input_re2 = np.array(input_re2, dtype=np.float32)
    input_x = tf.keras.preprocessing.sequence.pad_sequences(input_x, max_len, padding='post', truncating='post')
    input_ner = tf.keras.preprocessing.sequence.pad_sequences(input_ner, max_len, padding='post', truncating='post')
    position_mask = tf.keras.preprocessing.sequence.pad_sequences(position_mask, max_len, padding='post', truncating='post')
    return input_x, input_ner, input_re1, input_re2,  position_mask

# train_data = json.load(open('train_data_me.json', encoding='utf-8'))
# input_x, input_ner, input_re1, input_re2,  position_mask = get_input(train_data)
# print(train_data[0])
# print(input_x[0])
# print(input_ner[0])
# print(input_re1[0])
# print(input_re2[0])
'''
ner:预测subject/object
perdicate:预测头部关系矩阵(128*128)
'''
def get_input_so(data):
    input_x, input_ner, input_re = [], [], []
    for l in tqdm(range(32)):
        items = {}
        line = data[l]
        text = line['text'][:128]
        spo = line['spo_list']
        text2id = Token(text)
        for sp in spo:
            sp = (Token(sp[0]), sp[1], Token(sp[2]))
            subjectid = list_find(text2id, sp[0])
            objectid = list_find(text2id, sp[2])
            if subjectid != -1 and objectid != -1:
                key = (subjectid, subjectid + len(sp[0]))
                if key not in items:
                    items[key] = []
                items[key].append((objectid,
                                   objectid + len(sp[2]),
                                   predicate2id[sp[1]] + 1))
        if items:
            input_x.append(text2id)
            #seq_len.append(len(text2id))
            ner_s = np.zeros(len(text2id), dtype=np.int32)
            er_s = np.zeros((128, 128), dtype=np.int32)
            #mask_ = np.ones(len(text2id), dtype=np.int32)
            for j in items:
                ner_s[j[0]] = 1
                ner_s[j[0]+1:j[1]] = 2
                for k in items[j]:
                    ner_s[k[0]] = 1
                    ner_s[k[0]+1:k[1]] = 2
                    er_s[j[0]][k[0]] = k[2]
            #print(ner_s)
            input_ner.append(ner_s)
            input_re.append(er_s)
            #mask.append(mask_)


    #seq_len = np.array(seq_len, dtype=np.int32)
    input_re = np.array(input_re, dtype=np.int32)
    input_x = tf.keras.preprocessing.sequence.pad_sequences(input_x, max_len, padding='post', truncating='post')
    input_ner = tf.keras.preprocessing.sequence.pad_sequences(input_ner, max_len, padding='post', truncating='post')
    #mask = tf.keras.preprocessing.sequence.pad_sequences(mask, max_len, padding='post', truncating='post')
    return input_x, input_ner, input_re

# train_data = json.load(open('train_data_me.json', encoding='utf-8'))
# input_x, input_ner, input_re = get_input_so(train_data)
# print(train_data[0])
# print(input_x[0])
# print(input_ner[0])
# print(input_re[0][21])
