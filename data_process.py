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
    input_x, input_ner1, input_ner2, input_re1, input_re2, position_s, position_e = [], [], [], [], [], [], []
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
            ner_s1 = np.zeros(128, dtype=np.int32)
            ner_s2 = np.zeros(128, dtype=np.int32)
            for j in items:
                ner_s1[j[0]] = 1
                ner_s2[j[1]-1] = 1
            #print(ner_s)
            input_ner1.append(ner_s1)
            input_ner2.append(ner_s2)
            k1, k2 = np.array(list(items.keys())).T
            k1 = choice(k1)
            k2 = choice(k2[k2 >= k1])
            er_s1 = np.zeros((128, num_classes), dtype=np.float32)
            er_s2 = np.zeros((128, num_classes), dtype=np.float32)
            position_s.append(k1)
            position_e.append(k2 - 1)
            for j in items.get((k1, k2), []):
                er_s1[j[0]][j[2]] = 1
                er_s2[j[1] - 1][j[2]] = 1
            input_re1.append(er_s1)
            input_re2.append(er_s2)

    #seq_len = np.array(seq_len, dtype=np.int32)
    input_re1 = np.array(input_re1, dtype=np.int32)
    input_re2 = np.array(input_re2, dtype=np.int32)
    input_x = tf.keras.preprocessing.sequence.pad_sequences(input_x, max_len, padding='post', truncating='post')
    input_ner1 = np.array(input_ner1, dtype=np.int32)
    input_ner2 = np.array(input_ner2, dtype=np.int32)
    position_s = np.array(position_s, dtype=np.int32)
    position_e = np.array(position_e, dtype=np.int32)
    return input_x, input_ner1, input_ner2, input_re1, input_re2, position_s, position_e

# dev_data = json.load(open('./data_trans/dev_data_me.json', encoding='utf-8'))
# input_x, input_ner1, input_ner2, input_re1, input_re2, position_s, position_e = get_input(dev_data)
# print(input_ner1[1])
# print(input_ner1[2])
# print(input_ner2[1])
# print(input_ner2[2])
# input_ner1 = tf.one_hot(input_ner1[10], depth=2, dtype=tf.float32)
# print(input_ner1)
# input_ner1 = tf.argmax(input_ner1, axis=-1)
# input_ner1 = np.array(input_ner1, dtype=np.float32)
# print(np.where(input_ner1>0.5))
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
