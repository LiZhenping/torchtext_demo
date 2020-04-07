import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_load = pd.read_csv("train.tsv", sep='\t')
test = pd.read_csv("test.tsv", sep='\t')

# 建立验证集和训练集

train, val = train_test_split(data_load, test_size=0.2)

# 从新保存训练集和验证集

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

# 加载分词工具

token_en = spacy.load("en")


# 在此处分词
def tokenizer(text):
    return [tok.text for tok in token_en.tokenizer(text)]


# 定义Field对象 此处是 torchtext 的固定格式，定义通用的文本数据操作。
# Field 讲解见：https://www.cnblogs.com/helloeboy/p/9882467.html

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

# 在此处创建数据集

train, val = data.TabularDataset.splits(
    path='.', train='train.csv', validation='val.csv', format='csv', skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
)
test = data.TabularDataset(
    'test.tsv', format='tsv', skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)]
    )

# 在此处创建单词表
# 建立单词表：有两种方法，一种是使用训练集和测试集中的数据做单词表，另外一种是加载预训练模型单词表，该处为了方便对比，使用前一种。
TEXT.build_vocab(train)

# TEXT.build_vocab(train, vectors='glove.6B.100d')  # , max_size=30000)

"""
打印单词表方便对照
for i in range(len(TEXT.vocab.itos)):
    # s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    # s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(str(TEXT.vocab.itos[i]) + "\n")
file.close()
"""

# 在此处创建Iterator，在该处自动划分好batch，用于训练
# 具体参数间 https://www.jianshu.com/p/fef1c782d901

train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase),
                                 shuffle=True, device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase),
                               shuffle=True, device=DEVICE)

# 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
test_iter = data.Iterator(dataset=test, batch_size=128, train=False,
                          sort=False, device=DEVICE)

len_vocab = len(TEXT.vocab)
"""
打印创建的batch
batch = next(iter(train_iter))
print(batch)
print(batch.Phrase)
print(batch.Sentiment)
print(TEXT.vocab.freqs['A'])
print(TEXT.vocab.stoi['<pad>'])
print(TEXT.vocab.itos[1])

for i, v in enumerate(TEXT.vocab.stoi):
    if i == 5:
        break
    print(v)

"""

# 定义一个简单的 LSTM 模型


class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        self.lstm = nn.LSTM(100, 128, 3, batch_first=True)  # ,bidirectional=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        batch_size, seq_num = x.shape
        vec = self.embedding(x)
        out, (hn, cn) = self.lstm(vec)
        out = self.linear(out[:, -1, :])
        out = F.softmax(out, -1)
        return out


model = Enet()

"""
第二种词向量方式需要拷贝到embedding层
将前面生成的词向量矩阵拷贝到模型的embedding层
这样就自动的可以将输入的word index转为词向量
"""
# model.embedding.weight.data.copy_(TEXT.vocab.vectors)

model.to(DEVICE)

# 开始训练
# lr=0.000001

optimizer = optim.Adam(model.parameters())  # ,lr=0.000001)
n_epoch = 20

best_val_acc = 0

for epoch in range(n_epoch):

    for batch_idx, batch in enumerate(train_iter):
        data = batch.Phrase
        target = batch.Sentiment
        # 测试此处是否为必须
        count = 0
        for i in target:
            print(i)
            count = count + 1
            if count > 3:
                count = 0
                break

        # 该处代码作用
        #target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        #此处操作只能针对cpu数组的tensor 因此需要转化为cpu类型再转化为gpu tensor
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        # 转化为gpu tensor
        target = target.to(DEVICE)
        for i in target:
            print(i)
            count = count + 1
            if count > 3:
                count = 0
                break

        """
            tensor(2, device='cuda:0')
            tensor(2, device='cuda:0')
            tensor(0, device='cuda:0')
            tensor(3, device='cuda:0')
            tensor([0., 0., 1., 0., 0.])
            tensor([0., 0., 1., 0., 0.])
            tensor([1., 0., 0., 0., 0.])
            tensor([0., 0., 0., 1., 0.])        
        """

        target = target.to(DEVICE)
        # 此处的具体作用见备注
        data = data.permute(1, 0)
        """        
        测试传入数据
        count = 0
        for i in data:
            print(i)
            count = count+1
            if count >3:
                count = 0
                break
        print(data)
        # 这里是对读入数据进行转置变化，行列变化后，每行的数据会使一句话。data之前一句话是按照列，permute之后是按照行
        data = data.permute(1, 0)
        print(data)
        for i in  data:
            print(i)
            count = count+1
            if count >3:
                count = 0
                break
        """
        optimizer.zero_grad()

        out = model(data)
        # 求损失
        loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
        loss = loss.sum(-1).mean()

        loss.backward()
        optimizer.step()
        # 求准确率
        if (batch_idx + 1) % 200 == 0:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f' % (epoch, batch_idx, loss, acc))

    # 求经过多次训练后验证集准确率
    val_accs = []
    for batch_idx, batch in enumerate(val_iter):
        data = batch.Phrase
        target = batch.Sentiment
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1, 0)
        out = model(data)

        _, y_pre = torch.max(out, -1)
        acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
        val_accs.append(acc)

    acc = np.array(val_accs).mean()
    if acc > best_val_acc:
        print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
        torch.save(model.state_dict(), 'params.pkl')
        best_val_acc = acc
    print('val acc: %.4f' % (acc))




