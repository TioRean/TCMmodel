
# ------------------------------ 导包区 -----------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
import time
import logging
import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator
import io
import os

# ------------------------ 改良transformer模型 ---------------------------------

# 构建Embeddings类
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 构建PositionalEncoding类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model 词嵌入维度
        # dropout 代表Dropout层的置零比例
        # max_len 代表每个句子的最大长度
        super(PositionalEncoding, self).__init__()

        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，大小为max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len,).unsqueeze(1)

        # 定义一个变化矩阵 div_term 跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 将前面定义的变化矩阵进行奇数偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维tensor扩充为三维tensor
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步优化
        # 注册成buffer 后我们就可以在模型保存后重新加载时，将这个位置编码器和模型参数一同加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x代表文本序列的词嵌入表示
        # pe的编码过长，将第二个维度，即max_len对应的维度缩小成x句子的长度
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    # 将query 最后一维提取出来，代表词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力公式，将query和key的转置进行矩阵乘法(点乘)，然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))

    # 判断是否使用掩码张量
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 定义改良的softmax(quiet softmax)
    def softmax_1(x, dim=-1):
        # one_softmax = e**p / 1+ sum(e**p)

        # 计算每行的最大值
        # row_max = torch.max(x, dim=dim, keepdim=False)
        row_max = x.max(dim=dim)[0]
        row_max = row_max.unsqueeze(-1)
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        x = torch.sub(x, row_max)
        # 计算e的指数次幂
        x_exp = torch.exp(x)
        # 将x-exp相加
        x_sum = torch.sum(x_exp, dim=dim, keepdims=True)
        # 计算one_softmax
        res = torch.div(x_exp, 1 + x_sum)
        return res

    # 对scores的最后一个维度进行softmax操作
    p_attn = softmax_1(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 完成p_attn和value的乘法，并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    # 实现克隆函数，因为在多头注意力中，要用到多个结构相同的线性层
    # 需要使用clone函数将他们一同初始化到一个网络层列表对象中
    # module:代表要克隆的目标网络
    # N： module克隆的个数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制
class MultiHeadedAttendtion(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head 多头注意力机制中的头数
        # embedding_dim 代表词嵌入的维度
        # dropout: 进行Dropout操作时，置零的比例
        super(MultiHeadedAttendtion, self).__init__()

        # 要确认一个事实：多头数量head需要整除词嵌入的维度embbeding_dim
        assert embedding_dim % head == 0
        # 得到每个头获得词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        # 获得线形层，获得4个，分别是Q,K,V以及最终输出的线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张量
        self.attn = None
        # 初始化dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value时注意力机制的三个输入张量，mask代表掩码张量
        if mask is not None:
            # 使用squeeze 将掩码张量及逆行维度扩充，代表多头中第N个头
            mask = mask.unsqueeze(1)
        # 得到batch_size
        batch_size = query.size(0)

        # 使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行梯度和形状的改变
        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入到注意力层
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到的每个头的计算结果时4维张量，需要进行形状转换
        # 前面已经和1，2两个维度进行转置， 这里要转置回来
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后将x输入线性层列表中的最后一个线性层中进行处理，得到最终的多头注意力结构输出
        return self.linears[-1](x)


# 构建前馈全连接网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model 代表词嵌入的维度，同时也是两个线性层的输入维度和输出维度
        # d_ff 代表第一个线性层的输出维度，和第二个线性层的输入维度
        # dropout 经过Dropout层处理时，随机置零的比率
        super(PositionwiseFeedForward, self).__init__()

        # 定义两层全连接的线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x代表来自上一层的输出，
        # 将x送入第一个线性层网络->经过relu函数的激活->经历dropout层的处理->送入第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 构建规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        # features 代表词嵌入的维度
        # eps： 一个足够小的正数，用来在规范化计算公式的分母中，防止除零操作
        super(LayerNorm, self).__init__()

        # 初始化两个参数张量a2， b2， 用于对结果做规范化操作的计算
        # 将其用nn.Parameter 进行封装，代表他们也是模型中的参数
        self.a2 = nn.Parameter(torch.ones(feature))
        self.b2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        # x:代表上一层网络的输出
        # 对x进行最后一个维度的求均值操作，同时保持输出输入维度一致
        mean = x.mean(-1, keepdim=True)
        # 对x进行最后一个维度上的求标准差的操作，同时保持输出输入维度一致
        std = x.std(-1, keepdim=True)
        # 按照规范化公式进行计算并返回
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# 构建子层连接的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size: 代表词嵌入维度
        # dropout 进行Dropout 操作的置零比率
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        # x代表上一层传入的张量
        # sublayer 该子层连接中子层函数
        # 首先将x进行规范化，然后送入子层函数中，处理结果进入dropout层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))


# 构建编码器的类
class EncodeLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size: 词嵌入维度
        # self_attn 代表传入的多头子注意力子层实例化对象
        # feed_forward 前馈前连接层的实例化对象
        # dropout 置零比率
        super(EncodeLayer, self).__init__()

        # 将两个实例化对象和参数传入类中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        # 编码器层中有2个子层连接结构，使用clones函数进行操作
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # x代表上一层的传入张量
        # mask代表掩码张量
        # 首先让x经过第一个子层连接结构，内部包含多头注意力机制子层
        # 再让张量经过第二个子层链接结构，其中包含前馈全连接层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 构建编码器类 Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer: 代表编码器层
        # N：代表编码器中有layer层的个数
        super(Encoder, self).__init__()
        # 使用clones函数克隆N个编码器层
        self.layers = clones(layer, N)
        # 初始化一个规范化层，作用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # x: 代表上一层输出的张量
        # mask：代表掩码张量
        # 让x一次经历N个编码器的处理，最后再经过规范化层输出
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 构建解码器层类
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size 代表词嵌入维度
        # self_attn 代表多头自注意力机制的对象
        # src_attn 代表常规的注意力机制的对象
        # feed_forward 代表的前馈全连接层的对象
        # Dropout 代表Dropout的置零比率
        super(DecoderLayer, self).__init__()

        # 将参数传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        # 按照解码器层的结构图，使用clones函数克隆3个子层链接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # x： 代表上一层输入的张量
        # memory： 代表编码器的语义存储张量
        # source_mask：源数据的掩码张量
        # target_mask: 目标数据的掩码张量

        m = memory

        # 让x经历第一个子层，多头子注意力机制的子层
        # 采用target_mask，为了将解码时未来的信息进行遮掩，比如模型解码第二个字符，只能看见第一个字符信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 让x经历第二个子层，常规的注意力机制的子层，Q ！= K =V
        # 采用source_mask，为了遮蔽对结果信息无用的数据
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 让x经历第三个子层，即前馈全连接层
        return self.sublayer[2](x, self.feed_forward)


# 构建解码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        # layer：代表解码器层的对象
        # N：代表将layer进行拷贝的个数
        super(Decoder, self).__init__()
        # 利用clones函数克隆N个layer
        self.layers = clones(layer, N)
        # 实例化一个规范化层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        # x：代表目标数据的嵌入表示
        # memory：代表编码器的输出张量
        # source_mask：源数据的掩码张量
        # target_mask: 目标数据的掩码张量
        # 要将x一次经历所有的编码器层处理，最后同归规范化层
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 构建Generator类
class Generator(nn.Module):
    # 用以输出transformer模型的输出
    def __init__(self, d_model, vocab_size):
        # d_model 代表词嵌入的维度
        # vocab_size 代表词表的总大小
        super(Generator, self).__init__()
        # 定义一个线性层，作用是完成网络输出维度的变换
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


# 构建编码器-解码器结构类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        # encoder:编码器对象
        # decoder:解码器对象
        # source_embed:源数据的嵌入函数
        # target_embed:目标数据的嵌入函数
        # generator:输出部分类别生成器对象
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        # source:源数据
        # target:目标数据
        # source_mask:源数据的掩码张量
        # target_mask:目标数据的掩码张量
        return self.decode(self.encode(source, source_mask), source_mask,
                           target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        # memory代表经历编码器编码的输出张量
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    # source_vocab:源数据的词汇总数
    # target_vocab:目标数据的词汇总数
    # N:代表编码器和解码器堆叠的层数
    # d_model:词嵌入维度
    # d_ff:全连接层中变换矩阵的维度
    # head:多头注意力机制的头数
    # dropout:置零比率
    c = copy.deepcopy

    # 实例化一个多头注意力的类
    attn = MultiHeadedAttendtion(head, d_model)

    # 实例化一个前馈全连接层的网络对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化一个位置编码器
    position = PositionalEncoding(d_model, dropout)

    # 实例化模型model， 利用的是EncoderDecoder
    # 编码器结构中有2个子层，分别为attention层和前馈全连接层
    # 解码器结构中有3个子层，两个attention层和前馈全连接层
    # 本模型中，src不使用positional encoding，原因在于方证的顺序对辨证影响很小，且经过程序处理，已不是原本的顺序
    # trg使用positional encoding，原因在于，有必要将君药放在最前面，然后按药物重要程度、方剂既定的顺序、某药对下一个药的提示作用等因素排列
    # 以此来增加模型对trg预测的准确性
    model = EncoderDecoder(
        Encoder(EncodeLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 初始化整个模型中的参数， 判断参数的维度大于1， 将矩阵初始化成为一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# --------------------数据加载器及数据预处理--------------------------

# 定义数据加载器
class DataPipe:
    def __init__(self, FILE_PATH, vocabs=None, batch_size=20, batch_num=5, bucket_num=1):
        self.FILE_PATH = FILE_PATH
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.bucket_num = bucket_num
        if vocabs is not None:
            self.source_vocab, self.target_vocab = vocabs
        else:
            self.source_vocab, self.target_vocab = self.get_vocab()

    def get_datapipe(self):
        data_pipe = dp.iter.IterableWrapper([self.FILE_PATH])
        data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
        data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

        # 用data_pipe.map()方法将data.pipe映射成经过applyTransform方法的列表
        # data_pipe.map()将对data.pipe中的每个元素都进行applyTransform方法处理
        data_pipe = data_pipe.map(self.__applyTransform)

        # batch_size 每个batch的大小
        # batch_num 每个bucket有多少个batch
        # bucket_num 在pool中共有多少个bucket
        # sort_key 指定某个方法传入一个bucket然后对其进行排序
        data_pipe = data_pipe.bucketbatch(
            batch_size=self.batch_size, batch_num=self.batch_num, bucket_num=self.bucket_num,
            use_in_batch_shuffle=True, sort_key=self.__sort_bucket
        )

        # 将src及trg分为两个组，每个组有batch_size数目的src或trg
        data_pipe = data_pipe.map(self.__separateSourceTarget)

        # 添加padding
        data_pipe = data_pipe.map(self.__applyPadding)

        return data_pipe

    def get_vocab(self):
        # 获得vocab，index=0为source_vocab，index=1为target_vocab

        # 通过data_pipe传入数据，包括x=方证，y=药物
        data_pipe = dp.iter.IterableWrapper([self.FILE_PATH])
        data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
        data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
        # source（方证）的词表
        # 通过build_vocab_from_iterator 方法构建source的词表，
        # getTokens(data_pipe,0)表示通过getTokens实现的迭代器逐个输入方证
        # min_freq 列入词表的最低出现频次
        # specials特殊字符，'<pad>', '<sos>', '<eos>', '<unk> 的index分别为0，1，2，3
        source_vocab = build_vocab_from_iterator(
            self.__getTokens(data_pipe, 0),
            min_freq=1,
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        source_vocab.set_default_index(source_vocab['<unk>'])

        # target（药物）的词表
        target_vocab = build_vocab_from_iterator(
            self.__getTokens(data_pipe, 1),
            min_freq=1,
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        target_vocab.set_default_index(target_vocab['<unk>'])
        return source_vocab, target_vocab

    def __getTransform(self, vocab):
        text_transform = T.Sequential(
            # 给每个instance进行数字化（即Numericalize sentences using vocabulary）
            T.VocabTransform(vocab=vocab),
            # Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
            # 1 as seen in previous section
            T.AddToken(1, begin=True),
            # Add <eos> at end of each sentence. 2 because the index for <eos> in vocabulary is
            # 2 as seen in previous section
            T.AddToken(2, begin=False))
        return text_transform

    def __applyTransform(self, sequence_pair):
        # 定义将词元素转为根据vocab对应的index形式的方法
        return (
            self.__getTransform(self.source_vocab)(self.__tokenize(sequence_pair[0])),
            self.__getTransform(self.target_vocab)(self.__tokenize(sequence_pair[1])))

    def __tokenize(self, text):
        # 构建tokenizer（分词器），本模型构建时已完成分词，仅需通过‘， ’将不同元素转化成一个列表
        return text.split(', ')

    def __getTokens(self, data_iter, place):
        # 构建词表vocab
        # 通过迭代器获取Tokens
        # x为source（方证），y为target（药物）
        for x, y in data_iter:
            if place == 0:
                yield self.__tokenize(x)
            else:
                yield self.__tokenize(y)

    def __sort_bucket(self, bucket):
        # 构建产生batches的方法（这里使用bucketbatch）
        """
        Function to sort a given bucket. Here, we want to sort based on the length of
        source and target sequence.
        """
        return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

    def __separateSourceTarget(self, sequence_pairs):
        # 此时list(data_pipe)[0]格式为 [(X_1,y_1), (X_2,y_2), (X_3,y_3), (X_4,y_4)]
        # 我们希望将list(data_pipe)[0]的格式转化为((X_1,X_2,X_3,X_4), (y_1,y_2,y_3,y_4))
        sources, targets = zip(*sequence_pairs)
        return sources, targets

    # print(list(data_pipe)[0])

    def __applyPadding(self, pair_of_sequences):
        # 对序列进行padding
        # `T.ToTensor(0)`方法将序列转化为torch.tensor的格式且进行padding 这里的0是<pad>的在vocab中的index
        return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))


def getTransform(vocab):
    text_transform = T.Sequential(
        # 给每个instance进行数字化（即Numericalize sentences using vocabulary）
        T.VocabTransform(vocab=vocab),
        # Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        # Add <eos> at end of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False))
    return text_transform


# -------------------------训练模型及使用模型的工具---------------------------------

def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它最后两维形成一个方阵
    attn_shape = (1, size, size)
    # 然后使用np.ones方法向这个形状中添加1元素，形成上三角阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 最后将numpy类型转化为torch中的tensor，内部做一个1- 的操作。这个其实是做了一个三角阵的反转，subsequent_mask中的每个元素都会被1减。
    # 如果是0，subsequent_mask中的该位置由0变成1
    # 如果是1，subsequect_mask中的该位置由1变成0
    return torch.from_numpy(subsequent_mask) == 0


def data_generator(iterable):
    # 构建生成数据的data_generator方法，调用datapipe.get_datapipe方法，逐个导出数据的source和target
    for data in iterable:
        source = Variable(data[0], requires_grad=False)
        target = Variable(data[1], requires_grad=False)
        yield Batch(source, target)


class Batch:
    # Batch 实例化对象封装src，trg，src_mask，trg_mask
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # self.trg为用于进入模型计算的target
            # self.trg_y为用于计算loss时的target，与输出的output一同计算loss
            self.trg = trg[:, :-1]  # 通过前一个位置
            self.trg_y = trg[:, 1:]  # 预测后一个位置
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # 产生一个掩码张量用于遮掩padding及未来的数据
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute, vocabs, print_interval=300, is_print_instances=False):
    # run_epoch：为训练模型的方法
    # data_iter：用于传入数据的迭代器
    # model：即将被训练的模型
    # loss_compute：loss_compute的实例化对象
    # vocabs：词表，包括src及trg的词表
    # print_interval：打印间隔
    # is_print_instances：是否打印病例，是则每个print_interval打印当前batch的所有病例
    start_1 = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    # 通过vocab获取将数据从索引转为序列的方法
    src_itos = vocabs[0].get_itos()
    trg_itos = vocabs[1].get_itos()

    for i, batch in enumerate(data_iter):
        if torch.cuda.is_available():
            # 将数据传入cuda中
            batch.src = batch.src.cuda()
            batch.trg = batch.trg.cuda()
            batch.src_mask = batch.src_mask.cuda()
            batch.trg_mask = batch.trg_mask.cuda()
            batch.trg_y = batch.trg_y.cuda()
            batch.ntokens = batch.ntokens.cuda()

        # 获得模型的输出
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        # 通过out及batch.trg_y计算loss
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % print_interval == 1 and i != 1:
            elapsed = time.time() - start_1
            # 将srcs trgs分割成多个向量，每个向量代表一个instance，对preds进行贪婪解码
            srcs_tensor, preds_tensor, trgs_tensor = \
                __get_srcs_preds_trgs_tensor_list(model, batch.src, batch.src_mask, batch.trg)
            # 计算准确率
            accuracy = __count_accuracy(preds_tensor, trgs_tensor)
            if is_print_instances and vocabs is not None:
                    # 将srcs， preds， trgs经过vocab转为str并打印
                    for src, pred, trg in zip(srcs_tensor, preds_tensor, trgs_tensor):
                        print(f'src: {__idx_to_seq(src, src_itos)}')
                        print(f'pred: {__idx_to_seq(pred, trg_itos)}')
                        print(f'trg: {__idx_to_seq(trg, trg_itos)}')
                        print('-----------------------------------------------------------')
                    print('==========================================================================================')
            print("Epoch Step: %d || Loss: %f || Accuracy: %f || Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, accuracy, tokens / elapsed))
            print('==========================================================================================')
            start_1 = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    # get_stp_opt 的实例化对象
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # Update parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 计算每一个step的学习率
        # 每步学习率的计算公式   L = d_model**0.5 * min(step**(-0.5), step*warmup**(-1.5))
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    # 获取标准的优化器的实例化对象的方法
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    # 使用LabelSmoothing获得标签平滑对象
    # size 代表目标数据的词汇总数，即模型最后一层到张量的最后一维大小
    # padding_idx=0 表示不进行替换的维度
    # smoothing表示标签平滑程度， 如果原来标签表示值为1，则平滑后它的值域变为[1-smoothing，1+smoothing]
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # 相当于给target实现one-hot，然后算出x与target之间的loss
        # 如果label_smothing >0 则对该优化器进行正则化

        assert x.size(1) == self.size

        # 复制一个true_dist 即x
        true_dist = x.data.clone()

        # size为target的seq_len长度，即句子的长度，元素的个数
        true_dist.fill_(self.smoothing / (self.size - 2))

        # 相当于给target实现one-hot
        # 将one-hot中的1赋值为self.confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # one-hot中索引第0位的数据进行padding，赋值为0，相当于原本进行padding 的元素在one-hot上全部赋值为0
        true_dist[:, self.padding_idx] = 0

        # 将target中进行padding 的位置索引取出
        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            # 给true_dist进行mask操作
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    # 计算loss，更新参数（在train模式下）
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 输出概率最高的target
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # next_word == 2 是<eos>
        if next_word == 2:
            break
    return ys


def __idx_to_seq(indices, index_to_string):
    # idx_to_seq方法将索引转化为序列，即将数字化的药物转化为文字的药物
    # index_to_string 为source或target的vocab对象，然后获取get_itos()方法的实例化对象
    result = ''
    for idx in indices:
        str = index_to_string[idx]
        if str == '<eos>' or str == '<sos>' or str == '<pad>':
            continue
        else:
            result += " " + str
    return result


def __count_accuracy(preds_tensor, trgs_tensor):
    # 输入预测值的tensor及目标值的tensor，得出二者重合的元素与trgs所有元素个数和的比值，即准确度
    # 将tensor转为集合，去除padding及其他重复元素
    preds_scalar = [set(pred.tolist()) for pred in preds_tensor]
    trgs_scaler = [set(trg.tolist()) for trg in trgs_tensor]
    same_element_num = 0
    trg_element_total_num = 0
    for pred, trg in zip(preds_scalar, trgs_scaler):
        pred = list(filter(lambda x: x!=0 and x!=1 and x!=2, pred))
        trg = list(filter(lambda x: x!=0 and x!=1 and x!=2, trg))
        same_element_num += len(set(pred) & set(trg))
        trg_element_total_num += len(trg)
    return same_element_num / trg_element_total_num


def __get_srcs_preds_trgs_tensor_list(model, srcs, src_masks, trgs):
    # 本方法获取srcs_tensor、preds_tensor及trgs_tensor，每组类型均为list，每个list内包含batch_size个tensor，每个tensor代表一个instance
    # model：当前使用的模型，即transformer
    # srcs、src_mask、trgs都为Batch中的对应值，分别为源数据，源数据mask，目标数据
    # 通过贪婪解码获取预测值preds_tensor
    preds_tensor = [(greedy_decode(model, src.unsqueeze(0), mask.unsqueeze(0), max_len=20, start_symbol=1)).squeeze(0)
                    for src, mask in zip(list(srcs), list(src_masks))]

    # 将srcs(tensor)及trgs(tensor)以第0维分割，获取每个病例的tensor，装进一个list中
    srcs = torch.split(srcs, 1, 0)
    srcs_tensor = [src.squeeze(0) for src in srcs]
    trgs = torch.split(trgs, 1, 0)
    trgs_tensor = [trg.squeeze(0) for trg in trgs]

    return srcs_tensor, preds_tensor, trgs_tensor


def run(model, loss_compute, train_dp, val_dp, vocabs, save_dict=None, epochs=5, epoch_start=0):
    # model：代表将要训练的模型
    # loss_compute：代表使用的损失计算方法
    # train_dp, val_dp：为传输数据的迭代器
    # vocabs：src及trg的词表
    # save_dict：保存路径的文件夹
    # epochs：代表模型需要训练的轮次数
    # epoch_start：迭代开始时的epoch，用于模型再训练
    assert epoch_start >= 0 and epoch_start < epochs
    smallest_val_loss = 100000000

    for epoch in range(epoch_start, epochs):
        print(f'---------------------当前第 {epoch}/{epochs} epoch--------------------')
        start = time.perf_counter()

        # 进入训练模式，所有的参数将会被更新
        print('当前为训练模式：')
        model.train()
        train_loss = run_epoch(data_generator(train_dp), model, loss_compute, vocabs,
                               print_interval=500)
        # 进入评估模式，所有的参数固定不变
        print('\n当前为验证模式: ')
        model.eval()
        val_loss = run_epoch(data_generator(val_dp), model, loss_compute, vocabs,
                    print_interval=200, is_print_instances=True)
        print('==========================================================================================')
        print(f'本轮epoch结果: train_loss: {train_loss}, val_loss: {val_loss}')
        print(f'本轮epoch用时：{format((time.perf_counter() - start) / 3600, ".2f")}小时')
        print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

        # 保存模型
        if save_dict is not None:
            if val_loss < smallest_val_loss:
                save_model(save_dict, epoch + 1, transformer_model, loss_compute.opt.optimizer, train_loss, vocabs)
                smallest_val_loss = val_loss
                print('\n\n')
            else:
                print('\n\n')
                continue


def save_model(path_dir, epoch, model, optimizer, loss=None, vocabs=None):
    # 保存模型，path_dict：文件夹名称，epoch：当前迭代次数，model：模型，optimizer：优化器， loss：当前的loss值
    # vocabs 词表包括src_vocab及trg_vocab

    if not os.path.exists('./' + path_dir):
        os.mkdir('./' + path_dir)
    torch.save(model, './' + path_dir + 'model.pth')
    torch.save(model.state_dict(), './' + path_dir + 'model_weights.pth')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocabs': vocabs,
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, './' + path_dir + '/checkpoint.pth')
    print('模型及数据保存成功')


class ModelLoader:
    # 封装model_loader进行读取模型数据
    def __init__(self, model_path, checkpoint_path, loss_compute):
        # model_path：模型路径
        # checkpoint_path：checkpoint路径
        # loss_compute：loss_compute的实例化对象
        checkpoint = torch.load(checkpoint_path)
        self.model = \
            torch.load(model_path, map_location=torch.device('cuda')) if torch.cuda.is_available() else torch.load(
                model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.loss_compute = loss_compute
        self.loss_compute.opt.optimizer.load_state_dict(checkpoint['optimizer'])
        self.vocabs = checkpoint['vocabs']

    def run_model(self, train_dp, val_dp, save_dict, epochs=96):
        # run_model用于运行模型，内置run方法
        run(self.model, self.loss_compute, train_dp, val_dp, self.vocabs, save_dict, epochs, self.epoch)

    def run_test(self, data_pipe, is_save=False):
        # run_validation用于查看模型效果，内置print_test_result方法
        print_test_result(data_generator(data_pipe), self.model, self.vocabs, is_save=is_save)


# ---------------------模型训练后相关工具--------------------------------

def get_dosage(raw_prescription):
   # 原本无剂量的药方获取对应的剂量
   # raw_prescription为无剂量的药方，以字符串形式传入
    dosage_table= {'桂枝':10, '桂枝+':10, '桂枝++':10, '麻黄':10, '麻黄+':10, '细辛':5, '杏仁':10, '防风':10,'防己':10,
    '荆芥':10, '羌活':10, '独活':10, '大黄':10, '大黄+':10, '芒硝':10, '柴胡':20, '柴胡+':20, '黄芩':15, '桔梗':10, '瓜蒌皮':20,
    '天花粉':20, '黄连':10, '栀子':10, '豆豉':10, '半夏':15, '白芍':10, '白芍+':10, '赤芍':10, '赤芍+':10, '当归':15, '五味子':5,
    '木通':10, '茯苓':30, '白术':15, '黄芪':30, '党参':10, '党参+':10, '桃仁':10, '鸡内金':15, '蝉衣':10, '全蝎':5, '木瓜':15,
    '茅根':15, '贝母':10, '茵陈':30, '葛根':30, '葛根+':30, '石膏':30, '知母':10, '干姜':10, '附子':10, '附子+':10, '枳实':10,
    '枳实+':10, '厚朴':10, '厚朴+':10, '山药':30, '生姜':15, '生姜+':10, '大枣':15, '甘草':10, '龙骨':15, '牡蛎':15}

    medicine_list = raw_prescription.split(' ')
    content = {}
    # 获取所有药物及计量，以键值对形式存入字典中
    for item in medicine_list:
        content[item] = dosage_table.get(item)
    # 将带有XX+及XX++的药物计量转加到XX的剂量中
    for key in content.keys():
        if '+' in key:
            value = content[key]
            key_stripe = key.replace('+', '')
            content[key_stripe] = content[key_stripe] + value
            del content[key]
    # 将字典content转为特定格式的字符串输出
    result = ''
    for key in content.keys():
        result = result + key + raw_prescription(content[key]) + 'g '
    return result


def src_to_trg(model, source, numerize, target_itos):
    # src_to_trg方法将源数据经过模型输出结果（预测的target）
    # 给src进行数字编码
    # 将转化过的source序列转为tensor模式，传入Variable中作为参数
    source_applyTransform = numerize(source)
    src_tensor = torch.LongTensor([source_applyTransform])
    # 无需使用mask，构造与src_tensor形状一致的全1 tensor
    source_mask = torch.ones(src_tensor.shape)
    # 将数据传入cuda中计算
    if torch.cuda.is_available():
        src_tensor = src_tensor.cuda()
        source_mask = source_mask.cuda()
    # 进入评估模式，参数不更新
    model.eval()
    # 使用贪婪解码，即输出最大概率的target，设定解码的最大长度max_len等于20，起始数字的标志默认等于1
    result = greedy_decode(model, src_tensor, source_mask, max_len=20, start_symbol=1)
    # 为result降维处理
    result = result.squeeze(0)
    result = __idx_to_seq(result, target_itos)
    return result


def src_tensor_to_target(model, source_tensor):
    # src_tensor_to_target方法将以张量为形式的源数据经过模型输出预测的target
    # 设置掩码张量
    source_mask = (source_tensor != 0).unsqueeze(-2)
    # 进入评估模式，参数不更新
    model.eval()
    # 使用贪婪解码，即输出最大概率的target，设定解码的最大长度max_len等于20，起始数字的标志默认等于1
    result = greedy_decode(model, source_tensor, source_mask, max_len=20, start_symbol=1)
    # 为result降维处理
    result = result.squeeze(0)
    result = __idx_to_seq(result, train_trg_vocab)
    return result


def print_test_result(data_iter, model, vocabs, is_save=False, save_path='./test_result.txt'):
    # print_test_result 用于打印测试数据集的结果
    # data_iter：用于传入数据的迭代器
    # model：被训练的模型
    # vocabs: 词表，包含src和trg的词表
    src_itos = vocabs[0].get_itos()
    trg_itos = vocabs[1].get_itos()
    total_accuracy = 0
    count = 0
    print('==========================================================================================')
    for i, batch in enumerate(data_iter):
        if torch.cuda.is_available():
            # 将数据传入cuda中
            batch.src = batch.src.cuda()
            batch.trg = batch.trg.cuda()
            batch.src_mask = batch.src_mask.cuda()
            batch.trg_mask = batch.trg_mask.cuda()
            batch.trg_y = batch.trg_y.cuda()

        # 将srcs trgs分割成多个向量，每个向量代表一个instance，对preds进行贪婪解码
        srcs_tensor, preds_tensor, trgs_tensor = \
            __get_srcs_preds_trgs_tensor_list(model, batch.src, batch.src_mask, batch.trg)
        # 计算准确率
        accuracy = __count_accuracy(preds_tensor, trgs_tensor)
        total_accuracy += accuracy
        count += 1
        # 将srcs， preds， trgs经过vocab转为str并打印
        print('本批的准确率：{:.2f}%'.format(accuracy * 100))
        for src, pred, trg in zip(srcs_tensor, preds_tensor, trgs_tensor):
            print('-----------------------------------------------------------')
            src = __idx_to_seq(src, src_itos)
            pred = __idx_to_seq(pred, trg_itos)
            trg = __idx_to_seq(trg, trg_itos)
            print(f'输入方证: {src}')
            print(f'预测方药: {pred}')
            print(f'目标方药: {trg}')
            if is_save is True:
                with open(save_path, 'a') as f:
                    f.write(f'输入方证: {src} \n')
                    f.write(f'预测方药: {pred} \n')
                    f.write(f'目标方药: {trg} \n\n')
        print('==========================================================================================')
    print('总准确率：{:.2f}%'.format(total_accuracy / count * 100))


def get_tsv(model, dir, vocab):
    # 用于embedding projector所需要的vectors及metas
    # model 待获取数据的模型
    # dir 存储的文件夹名称
    # vocab 词典
    if not os.path.exists('./' + dir):
        os.mkdir('./' + dir)
    weights = model.src_embed[0].lut.cpu().weight.detach().numpy()
    path_src_vectors = './' + dir + 'src_vecs.tsv'
    path_src_metas = './' + dir + 'src_meta.tsv'
    out_v = io.open(path_src_vectors, 'w', encoding='utf-8')
    out_m = io.open(path_src_metas, 'w', encoding='utf-8')
    for word_num in range(4, len(vocab)):
        word = vocab.lookup_token(word_num)  # index_word:{1:"OOV", 2:"the", ……}
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()


# ---------------------设置模型参数及模型构建--------------------------------
# 包括data， model， loss_fn， optimizer

# 获取训练集 验证集的数据集。   在run()函数中用data_generator调用datapipe产生source及target逐个获取
# 实例化datapipe.get_datapipe  有train 和val 或test
train_data_path = './train_data.txt'
train_bz = 20
train_datapipe = DataPipe(train_data_path, batch_size=train_bz, batch_num=5)
train_dp = train_datapipe.get_datapipe()
# 通过vocab方法获得不重复vocab及vocab总数
train_vocabs = train_datapipe.get_vocab()
train_src_vocab, train_trg_vocab = train_vocabs
ntokens_source = len(train_src_vocab)
ntokens_target = len(train_trg_vocab)
# 获取源数据数字化的实例化对象
numerize = getTransform(train_src_vocab)
# 获取vocab.get_itos()对象，即将index转为string
src_itos = train_src_vocab.get_itos()
trg_itos = train_trg_vocab.get_itos()

val_data_path = './test.txt'
val_bz = 10
val_datapipe = DataPipe(val_data_path, vocabs=train_vocabs , batch_size=val_bz, batch_num=5)
val_dp = val_datapipe.get_datapipe()

# 实例化transformer模型
# 词嵌入大小
emsize = 512
# 前馈全连接层的节点数
nhid = 2048
# 编码器层的数量
nlayer = 8
# 多头注意力机制的头数
nhead = 8
# 置零比率
dropout = 0.1

# 将参数输入到TransformerModel中
transformer_model = make_model(source_vocab=ntokens_source, target_vocab=ntokens_target, N=nlayer,
                               d_ff=nhid, head=nhead, dropout=dropout)

# 使用get_std_opt 获得模拟器优化器
model_optimizer = get_std_opt(transformer_model)

# 获取标签平滑化方法对象(target进行one-hot编码及标签平滑化)
smoothing = 0.1
criterion = LabelSmoothing(size=ntokens_target, padding_idx=0, smoothing=smoothing)

# 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
loss_compute = SimpleLossCompute(transformer_model.generator, criterion, model_optimizer)

# 将模型转入cuda中运行
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logging.warning('数据传入cuda中计算')
    transformer_model.cuda()
    criterion.cuda()

# 设置保存路径
save_dict = 'test_dict/'

# ----------------------------测试区A--------------------------------------

if __name__ == '__main__':
    print(f'train_data_path: {train_data_path}, batch_size: {train_bz}')
    print(f'val_data_path：{val_data_path}, batch_size: {val_bz}')
    print(f'save_dict: {save_dict}')
    print(f'model：transformer || loss_fn:KLDivLoss || optim:Adam ')
    print(f'emsize: {emsize}       || nhid：{nhid}         || nlayer: {nlayer}')
    print(f'nhead: {nhead}          || dropout: {dropout}')
    print(f'smoothing: {smoothing}')
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    print('\n\n')

    # run(transformer_model, loss_compute, train_dp, val_dp, train_vocabs, save_dict=save_dict, epochs=96)

# ----------------------------测试区B--------------------------------------

    path_model = './modelDict/model.pth'
    path_checkpiont = './modelDict/checkpoint.pth'

    ml = ModelLoader(path_model, path_checkpiont, loss_compute)
    # ml.run_model(train_dp, val_dp, save_dict, epochs=48)
    ml.run_test(val_dp, is_save=True)

    # get_tsv(ml.model, 'tsv_dir/', ml.vocabs[0])




