#! -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from data_process import Token, get_input, id2predicate
import json

train_data = json.load(open('./data_trans/train_data_me.json', encoding='utf-8'))
dev_data = json.load(open('./data_trans/dev_data_me.json', encoding='utf-8'))

num_class = 49
lr = 0.005
num_epochs = 20
batch_size = 16

class data_loader():
    def __init__(self):
        self.input_x, self.input_ner1, self.input_ner2, self.input_re1, self.input_re2, self.p_s, self.p_e = get_input(train_data)
        self.input_x = self.input_x.astype(np.int32)
        self.input_ner1 = self.input_ner1.astype(np.int32)
        self.input_ner2 = self.input_ner2.astype(np.int32)
        self.input_re1 = self.input_re1.astype(np.float32)
        self.input_re2 = self.input_re2.astype(np.float32)
        self.p_s = self.p_s.astype(np.int32)
        self.p_e = self.p_e.astype(np.int32)
        self.num_train = self.input_x.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices((self.input_x, self.input_ner1, self.input_ner2, self.input_re1, self.input_re2, self.p_s, self.p_e))
        self.db_train = self.db_train.shuffle(self.num_train).batch(batch_size, drop_remainder=True)

    def get_batch(self, batch_s):
        indics = np.random.randint(0, self.num_train, batch_s)
        return self.input_x[indics], self.input_ner1[indics], self.input_ner2[indics], self.input_re1[indics], self.input_re2[indics], self.p_s[indics], self.p_e[indics]


class Ner_model(tf.keras.Model):
    def __init__(self):
        super(Ner_model, self).__init__()
        self.char_embedding = tf.keras.layers.Embedding(4996, 64, mask_zero=True)
        self.bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.dense_1 = tf.keras.layers.Dense(1)
        self.dense_2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.char_embedding(inputs)
        mask = self.char_embedding.compute_mask(inputs)
        x_gru = self.bi_gru(x, mask=mask)
        x_1 = tf.nn.sigmoid(self.dense_1(x_gru))
        x_2 = tf.nn.sigmoid(self.dense_2(x_gru))
        return x_1, x_2, x_gru


class ER_model(tf.keras.Model):
    def __init__(self):
        super(ER_model, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(num_class)
        self.dense_2 = tf.keras.layers.Dense(num_class)
        self.average = tf.keras.layers.Average()
    def call(self, x_lstm, position_s, position_e):
        add_encode = np.zeros_like(x_lstm)
        for i, k in enumerate(position_s):
            gru_v = x_lstm[i, :, :]
            v_s = gru_v[k, :]
            v_e = gru_v[position_e[i], :]
            v_subject = self.average([v_s, v_e])
            add_encode[i, k, :] = v_subject
            add_encode[i, position_e[i], :] = v_subject

        x = x_lstm + add_encode
        output1 = tf.sigmoid(self.dense_1(x))
        output2 = tf.sigmoid(self.dense_2(x))
        return output1, output2


def loss_function(y_1, y_2, y_re1, y_re2, input_ner1, input_ner2, input_re1, input_re2):
    input_ner1 = tf.expand_dims(input_ner1, 2)
    loss_ner1 = tf.keras.losses.binary_crossentropy(y_true=input_ner1, y_pred=y_1)
    loss_ner1 = tf.reduce_sum(loss_ner1)

    input_ner2 = tf.expand_dims(input_ner2, 2)
    loss_ner2 = tf.keras.losses.binary_crossentropy(y_true=input_ner2, y_pred=y_2)
    loss_ner2 = tf.reduce_sum(loss_ner2)

    loss_re1 = tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true=input_re1, y_pred=y_re1), axis=-1, keepdims=True)
    loss_re1 = tf.reduce_sum(loss_re1)

    loss_re2 = tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true=input_re2, y_pred=y_re2), axis=-1, keepdims=True)
    loss_re2 = tf.reduce_sum(loss_re2)
    loss = (loss_ner1 + loss_ner2) + (loss_re1 + loss_re2)

    return loss, (loss_ner1 + loss_ner2), (loss_re1 + loss_re2)


class Extra_result(object):
    def __init__(self, text):
        self.text = text
    def call(self):
        result = []
        token = np.zeros(len(self.text))
        text2id = Token(self.text)
        token[0:len(text2id)] = text2id
        Model_ner = model_Ner
        Model_er = model_Er
        ner1, ner2, out_lm = Model_ner(np.array([token], dtype=np.int32))
        subjects = self.extra_sujects(ner1, ner2)
        for i, key in enumerate(subjects):
            ids1 = key[1]
            ids2 = key[2]
            re1, re2 = Model_er(out_lm, np.array([ids1], dtype=np.int32), np.array([ids2], dtype=np.int32))
            relationship = self.extra_er(key[0], re1, re2)
            result.extend(relationship)
        print(subjects)
        print(result)
        return result

    def extra_sujects(self, ner_1, ner_2):
        subject = []
        ner_1, ner_2 = np.where(ner_1[0] > 0.5)[0], np.where(ner_2[0] > 0.5)[0]
        if len(ner_1) > 0:
            for i in ner_1:
                j = ner_2[ner_2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _subject = self.text[i: j+1]
                    subject.append((_subject, i, j))
        return subject

    def extra_er(self, key, re1, re2):
        relationship = []
        o_re1, o_re2 = np.where(re1[0] > 0.5), np.where(re2[0] > 0.5)
        for _re1, c1 in zip(*o_re1):
            for _re2, c2 in zip(*o_re2):
                if _re1 <= _re2 and c1 == c2:
                    _object = self.text[_re1: _re2 + 1]
                    _predicate = id2predicate[c1]
                    relationship.append((key, _predicate, _object))
                    break
        return relationship


class Evaluate(object):
    def __init__(self):
        pass
    def reset(self,spo_list):
        xx = []
        for key in spo_list:
            xx.append((key[0], key[1], key[2]))
        return xx
    def evaluate(self, data):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in data[0:10]:
            extra_items = Extra_result(d['text'])
            R = set(extra_items.call())
            T = set(self.reset(d['spo_list']))
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C

model_Ner = Ner_model()
model_Er = ER_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model_Ner=model_Ner, model_Er=model_Er)
evaluate = Evaluate()
data_loader = data_loader()
best = 0.0

for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)

    num_batchs = int(data_loader.num_train / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x, input_ner1, input_ner2, input_re1, input_re2, position_s, position_e = data_loader.get_batch(batch_size)

        with tf.GradientTape() as tape:
            y_1, y_2, out_lstm = model_Ner(input_x) #预测ner
            y_re1, y_re2 = model_Er(out_lstm, position_s, position_e)
            loss, loss1, loss2 = loss_function(y_1, y_2, y_re1, y_re2, input_ner1, input_ner2, input_re1, input_re2)
            if (batch_index+1) % 500 == 0:
                print("batch %d: loss %f: loss1 %f: loss2 %f" % (batch_index+1, loss.numpy(), loss1.numpy(), loss2.numpy()))

        variables = (model_Ner.variables + model_Er.variables)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    F, P, R = evaluate.evaluate(dev_data)
    print('测试集:', "F %f: P %f: R %f: " % (F, P, F))
    if round(F, 2) > best and round(F, 2) > 0.50:
        best = F
        print('saving_model')
        #model.save('./save/Entity_Relationshaip_version2.h5')
        checkpoint.save('./save/Entity_Relationship/version1_checkpoints.ckpt')