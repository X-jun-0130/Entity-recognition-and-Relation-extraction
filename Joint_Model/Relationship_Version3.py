'''
使用双向GRU
F值最高81.8%
缺陷：无法进行实体重叠的关系抽取
'''

#! -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from data_process import Token, get_input_so, id2predicate
import json

train_data = json.load(open('./data_trans/train_data_me.json', encoding='utf-8'))
dev_data = json.load(open('./data_trans/dev_data_me.json', encoding='utf-8'))

len_char = 4996
char_dim = 128
num_class = 50
lr = 0.005
num_epochs = 20
batch_size = 16
dropout = 0.5

class data_loader():
    def __init__(self):
        self.input_x, self.input_ner, self.input_re = get_input_so(train_data)
        self.input_x = self.input_x.astype(np.int32)
        self.input_ner = self.input_ner.astype(np.int32)
        self.input_re = self.input_re.astype(np.int32)
        self.num_train = self.input_x.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices((self.input_x, self.input_ner, self.input_re))
        self.db_train = self.db_train.shuffle(self.num_train).batch(batch_size, drop_remainder=True)

    def get_batch(self, batch_s):
        indics = np.random.randint(0, self.num_train, batch_s)
        return self.input_x[indics], self.input_ner[indics], self.input_re[indics]
'''
epoch20, 最大F=81.1
'''
class ER_model(tf.keras.Model):
    def __init__(self):
        super(ER_model, self).__init__()
        self.char_embedding = tf.keras.layers.Embedding(4996, 64, mask_zero=True) #
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.dense_left = tf.keras.layers.Dense(100, use_bias=False)
        self.dense_right = tf.keras.layers.Dense(100, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(num_class)

#对子类化的模型或层的‘call’方法中，掩码不能被自动传播，所以你需手动将掩码参数传递任何需要它的层。
    def call(self, inputs):
        embedding = self.char_embedding(inputs)
        mask = self.char_embedding.compute_mask(inputs)
        encode_input = self.bi_lstm(embedding, mask=mask)
        left = self.dense_left(encode_input)
        right = self.dense_right(encode_input)
        outer_sum = broadcasting(left, right)
        output = tf.tanh(outer_sum)
        output = self.dropout(output)
        output = self.dense(output)
        output = tf.sigmoid(output)
        return output

def broadcasting(left, right):
    left = tf.transpose(left, perm=[1, 0, 2])
    left = tf.expand_dims(left, 3)
    right = tf.transpose(right, perm=[0, 2, 1])
    right = tf.expand_dims(right, 0)
    B = left + right
    B = tf.transpose(B, perm=[1, 0, 3, 2])
    return B

def loss_function(re_pred, input_red):

    input_re_onehot = tf.one_hot(input_red, depth=num_class, dtype=tf.float32)
    loss_re = tf.keras.losses.binary_crossentropy(y_true=input_re_onehot, y_pred=re_pred)
    loss_re = tf.reduce_sum(loss_re)
    loss = (loss_re)
    return loss


class Extra_result(object):
    def __init__(self, text, spo_list):
        self.text = text
        self.spo =spo_list
    def call(self):
        result = []
        token = np.zeros(len(self.text))
        text2id = Token(self.text)
        token[0:len(text2id)] = text2id
        Model_er = model_Er
        subjects = self.extra_sujects()
        re = Model_er(np.array([token], dtype=np.int32))
        relationship = self.extra_er(subjects, re)
        # print(subjects)
        print(relationship)
        result.extend(relationship)
        return result

    def extra_sujects(self):
        subject = []
        subject_ = []
        for key in self.spo:
                subject_.append(key[0])
                subject_.append(key[2])

        subject_ = list(set(subject_))
        for key in subject_:
            id = self.text.index(key)
            subject.append((key, id))
        return subject

    def extra_er(self, subjects, re):
        position = [key[1] for key in subjects]
        subjects_ = [key[0] for key in subjects]
        re = re[0]

        relationship = []
        re = tf.argmax(re, axis=-1)

        length = re.shape[0]
        for k in range(length):
            for i, key in enumerate(list(np.array(re[k]))):
                if int(key) > 0:
                    if k in position and i in position:
                        subject = subjects_[position.index(k)]
                        object = subjects_[position.index(i)]
                        predicate = id2predicate[key - 1]
                        relationship.append((subject, predicate, object))
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
        for d in data[0:100]:
            extra_items = Extra_result(d['text'], self.reset(d['spo_list']))
            R = set(extra_items.call())
            T = set(self.reset(d['spo_list']))
            A += len(R & T)#抽取正确数量
            B += len(R) #抽取数量
            C += len(T)#原正确数量
        return (2 * A / (B + C)), (A / B), (A / C)

#建立模型
model_Er = ER_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#保存模型
# checkpoint_dir = './save/Entity_Relationshaip_version2_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model_Er=model_Er)

evaluate = Evaluate()
data_loader = data_loader()
best = 0.0

for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    num_batchs = int(data_loader.num_train / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x, input_ner, input_re = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:

            y_re = model_Er(input_x) #预测关系
            loss = loss_function(y_re, input_re)
            if (batch_index+1) % 100 == 0:
                print("batch %d: loss %f" % (batch_index+1, loss.numpy()))

        variables = (model_Er.variables)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    F, P, R = evaluate.evaluate(dev_data)
    print('测试集:', "F: %f, P: %f, R: %f" % (F, P, F))
    if round(F, 2) > best and round(F, 2) > 0.50:
        best = F
        print('saving_model')
        #model.save('./save/Entity_Relationshaip_version2.h5')
        checkpoint.save('./save/Relationship/version3_checkpoints.ckpt')
