#! -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from data_process import Token, get_input_so, id2predicate
import json

train_data = json.load(open('./data_trans/train_data_me.json', encoding='utf-8'))
dev_data = json.load(open('./data_trans/dev_data_me.json', encoding='utf-8'))

len_char = 4996
char_dim = 128
label_class = 3
num_class = 50
lr = 0.005
num_epochs = 20
batch_size = 16
dropout = 0.5

# def load_pickle(file_path):
#     with open(file_path, 'rb') as f:
#         fileobj = pickle.load(f)
#     return fileobj
# embedding_matrix = load_pickle('./data_trans/embedding_matrix.pkl')

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


class Ner_model(tf.keras.Model):
    def __init__(self):
        super(Ner_model, self).__init__()
        self.char_embedding = tf.keras.layers.Embedding(4996, 64, mask_zero=True)
        self.bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(label_class)

    def call(self, inputs):
        x = self.char_embedding(inputs)
        mask = self.char_embedding.compute_mask(inputs)
        x_gru = self.bi_gru(x, mask=mask)
        x = self.dense(x_gru)
        x = self.dropout(x)
        x_ = tf.nn.softmax(x)
        return x_, x_gru


class ER_model(tf.keras.Model):
    def __init__(self):
        super(ER_model, self).__init__()
        #self.label_embedding = tf.keras.layers.Embedding(3, 64)
        # self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_left = tf.keras.layers.Dense(100, use_bias=False)
        self.dense_right = tf.keras.layers.Dense(100, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(num_class)

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
    #                               tf.TensorSpec(shape=(None, 128), dtype=tf.float32)])
    def call(self, encode_input):
        #label_embedding = self.label_embedding(ner)
        #mask = self.label_embedding.compute_mask(ner)
        #encode_input_hidden_size = encode_input.shape[-1]
        # u_a = tf.Variable("u_a", [encode_input_hidden_size + 64, hidden_size_n1])
        # w_a = tf.Variable("w_a", [encode_input_hidden_size + 64, hidden_size_n1])
        # v = tf.Variable("v", [hidden_size_n1, num_class])
        # b_s = tf.Variable("b_s", [hidden_size_n1])
        # print(u_a.shape)
        #encode_input = tf.concat([encode_input, label_embedding], axis=-1)
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


# def Mask(inputs):
#     mask = tf.math.logical_not(tf.math.equal(inputs, 0))
#     mask = tf.cast(mask, tf.float32)
#     # mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.keras.backend.greater(tf.expand_dims(x,2), 0), tf.float32))(inputs)
#     return mask

def loss_function(ner, re_pred, input_nerd, input_red):
    ner_one_hot = tf.one_hot(input_nerd, depth=3, dtype=tf.float32)
    loss_ner = tf.keras.losses.categorical_crossentropy(y_true=ner_one_hot, y_pred=ner)
    loss_ner = tf.reduce_sum(loss_ner)

    input_re_onehot = tf.one_hot(input_red, depth=num_class, dtype=tf.float32)
    loss_re = tf.keras.losses.binary_crossentropy(y_true=input_re_onehot, y_pred=re_pred)
    loss_re = tf.reduce_sum(loss_re)

    loss = (loss_ner + loss_re)
    return loss, loss_ner, loss_re


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
        ner, out_lm = Model_ner(np.array([token], dtype=np.int32))
        subjects, ner_list = self.extra_sujects(ner)
        re = Model_er(out_lm)
        relationship = self.extra_er(subjects, re)
        # print(subjects)
        # print(relationship)
        result.extend(relationship)
        return result

    def extra_sujects(self, ner_label):
        ner = ner_label[0]
        ner = tf.round(ner)
        ner = [tf.argmax(ner[k]) for k in range(ner.shape[0])]
        ner = list(np.array(ner))
        ner.append(0)#防止最后一位不为0
        text_list = [key for key in self.text]
        subject = []
        for i, k in enumerate(text_list):
            if int(ner[i]) == 0 or int(ner[i]) == 2:
                continue
            elif int(ner[i]) == 1:
                ner_back = [int(j) for j in ner[i + 1:]]
                if 1 in ner_back and 0 in ner_back:
                    indics_1 = ner_back.index(1) + i
                    indics_0 = ner_back.index(0) + i
                    subject.append((''.join(text_list[i: min(indics_0, indics_1) + 1]), i))
                elif 1 not in ner_back:
                    indics = ner_back.index(0) + i
                    subject.append((''.join(text_list[i:indics + 1]), i))
        return subject, ner[:-1]

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
            extra_items = Extra_result(d['text'])
            R = set(extra_items.call())
            T = set(self.reset(d['spo_list']))
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C

#建立模型
model_Ner = Ner_model()
model_Er = ER_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#保存模型
# checkpoint_dir = './save/Entity_Relationshaip_version2_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model_Ner=model_Ner, model_Er=model_Er)

evaluate = Evaluate()
data_loader = data_loader()
best = 0.0

for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)

    num_batchs = int(data_loader.num_train / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x, input_ner, input_re = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_ner, encode_gru = model_Ner(input_x) #预测ner
            y_re = model_Er(encode_gru) #预测关系
            # mask_ner = Mask(input_x)
            # mask_re = Mask(input_re)
            loss, loss1, loss2 = loss_function(y_ner, y_re, input_ner, input_re)
            if (batch_index+1) % 100 == 0:
                print("batch %d: loss %f: loss1 %f: loss2 %f" % (batch_index+1, loss.numpy(), loss1.numpy(), loss2.numpy()))

        variables = (model_Ner.variables + model_Er.variables)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    #f, p, r = evaluate.evaluate(train_data)
    F, P, R = evaluate.evaluate(dev_data)
    #print('训练集:', "f %f: p %f: r %f: " % (f, p, r))
    print('测试集:', "F %f: P %f: R %f: " % (F, P, F))
    if round(F, 2) > best and round(F, 2) > 0.50:
        best = F
        print('saving_model')
        #model.save('./save/Entity_Relationshaip_version2.h5')
        checkpoint.save('./save/Entity_Relationship/version2_checkpoints.ckpt')
