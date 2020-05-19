实体关系抽取是信息抽取任务中非常基础且必要的工作。实体关系主要有一对一、多对一、多对多等。今天从实践的角度介绍一下实体关系抽取的相关工作。仅为了简单介绍实体关系抽取相关的实践过程，模型我没有进行深度调优，故不适用实际生产中。仅在此介绍下方法，模型主要结构使用的双向GRU网络， 以及BERT。![Joint entity recognition and relation extraction as a multi-head selection problem](https://upload-images.jianshu.io/upload_images/12779090-f6b57e875738d096.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
模型依据论文“Joint entity recognition and relation extraction as a multi-head selection problem”进行改造而成，舍弃了Label Embeddings部分；没有使用CRF层进行实体的识别，主要是没有发现与TF2.0搭配的CRF库，进而用Softmax代替。 

## 数据来源
百度2019语言与智能技术竞赛信息抽取  

## 模型要求
Tensorflow-gpu=2.0.0
transformers  

# 一、实体关系联合抽取
目的是提取三元组[subject, predicate,object]；
基本简单再简单的原则，实体标签{O:0, B:1, I:2}, 进行subject,object提取；
关系部分，是一个多头选择策略(本文主要也是介绍这种方法)，对任一序列，构建[sequence_length, sequence_length]数组，Subject头位置[第N行]对应Object头位置[第M列]，该处数字表示第C种关系。  

## 模型一
使用双向GRU构建实体关系联合抽取模型。
使用字向量，首先制作字典，选取前4996个字；
构建模型输入数据，最大长度128，构建文本token，实体label， 关系label；
双向GRU输出+softmax提取实体；双向GRU输出+sigmoid提取关系。

模型使用64000条训练数据，最后对测试集前100进行验证，最大F1值是51%;
主要原因是单靠双向GRU的学习能力不够。  

## 模型二
使用BERT构建实体关系联合抽取模型。
构建模型输入数据，最大长度128，BERT输入又三部分构成[文本token, mask_token, segment_token]，实体label， 关系label；
BERT输出+softmax提取实体；BERT输出+sigmoid提取关系。

模型使用3000条训练数据，最后对测试集前100进行验证，最大F1值是81.8%;
模型太重，没有取过多数据训练，应该还可以继续提高的。

在模型二的基础上增加label_embeddings层，但模型并没有获得提升，F1值78.9%；猜测原因可能是实体标签(原论文里面包含标签的类型信息)过于简单，损失的信息过多，如此将label_embedding与bert输出结合，并没有起到作用。

# 二、关系抽取
## 模型一
不进行实体提取，当做已有实体直接进行关系预测。
使用双向GRU构建关系抽取模型。
使用字向量，首先制作字典，选取前4996个字；
构建模型输入数据，最大长度128，构建文本token，实体label， 关系label；双向GRU输出+sigmoid提取关系。

模型使用64000条训练数据，最后对测试集前100进行验证，关系提取最大F1值是81.8%，跟模型一对比，也验证了单靠双向GRU进行联合。  

## 模型二
使用BERT构建关系抽取模型。
构建模型输入数据，最大长度128，BERT输入又三部分构成[文本token, mask_token, segment_token]，实体label， 关系label；
BERT输出+sigmoid提取关系。

模型使用3000条训练数据，最后对测试集前100进行验证，模型没有跑完，前10个epoch时，模型F1已经超过85%。  

# 总结
本文的目的就是让大家了解下多头选择策略下的关系抽取问题。
  
