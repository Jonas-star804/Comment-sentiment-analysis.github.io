## Background
When shopping offline, people have become accustomed to checking the merchant's reviews on websites such as Meituan, Koubei and Ctrip to determine whether they are buying at the store.
Each merchant has a large amount of comment information on these apps, some positive and some negative. A smart APP automatically classifies user reviews.
This project uses the hotel evaluation data of Ctrip APP and combines natural language processing technology and machine learning methods to establish a natural language sentiment analysis model.
The model can automatically analyze and judge the emotion of user comments and determine whether the comments are positive or negative.

## Tools
- Text preprocessing;
- BiLstm modeling;
- TorchText tool is used;
- Flask framework deploys the model and opens the interface;
- Queue calls interfaces to categorize emotions;

## Process

### Configure (configure.py)


```
import torch

class BasicConfigs():
    #数据存放参数
    neg='data/neg'   # Negative sample directory
    pos='data/pos'   # Positive sample directory
    data_path='data' # Directory for storing partitioned data
    text_vocab_path='model_storage/text.vocab'#文本词典存放目录
    label_vocab_path ='model_storage/label.vocab'#标签词典存放目录
    stop_word_path='data/stopword.txt'#停用词文件路径
    #词向量参数
    embedding_loc = 'data/sgns.wiki.word'#词向量文件路径
    #模型训练参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'#设备
    lr = 0.001#学习率
    dropout_rate = 0.5#随机失活比例
    train_embedding = True#是否训练词嵌入向量
    batch_size = 64#批次大小
    alpha=0.0001#L2惩罚项系数
    #textcnn配置参数
    kernel_sizes = [3, 4, 5]  # 3个 conv1d的size
    num_channels = [100, 100, 100] # 卷积核个数
    # bilstm配置参数
    num_hiddens = 100#lstm神经元数
    num_layers = 1#lstm层数
    save_model_dir={#模型保存路径
        'textcnn':'model_storage/model_cnn.pt',
        'birnn':'model_storage/model_rnn.pt'
    }
```

### Text preprocessing (data_preprossing.py)
Since the positive and negative sample data we got were scattered in 3000 text files, we needed to read, sort out, segment, segment the data and create a dictionary.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from configs import BasicConfigs
from glob import glob
import jieba
import pickle
from torchtext.vocab import Vectors
from torchtext.data import Field, TabularDataset, BucketIterator
import re
bc = BasicConfigs()

def load_data_to_csv():
    #遍历文件夹 读取数据
    contents=[]
    for file in glob(bc.neg+"/*.txt"):#读取neg目录中的所有文件
        with open(file,'r',encoding='utf-8') as f:
            content=''.join([line.strip() for line in f.readlines()])#读取文本
            contents.append([content,'neg'])#添加标签
    for file in glob(bc.pos+"/*.txt"):#读取pos目录中的所有文件
        with open(file,'r',encoding='utf-8') as f:
            content = ''.join([line.strip() for line in f.readlines()])
            contents.append([content,'pos'])
    #打乱顺序并存储到train.csv,test.csv,val.csv
    #封装df
    df=pd.DataFrame(contents,columns=['text','label'])
    train,test=train_test_split(df,test_size=0.1,random_state=12)#数据分割
    train,val=train_test_split(train,test_size=0.2,random_state=12)#训练集再分割

# 文本清洗
def clearTxt(line):
    if line != '':
        line = line.strip()
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line

def my_cut(line):
    line=clearTxt(line)#清洗
    return jieba.lcut(line)#分词并返回中
    
def prepare_vocab(is_train=True):
    # 定义Field
    TEXT = Field(tokenize=my_cut)
    LABEL = Field(eos_token=None, pad_token=None, unk_token=None)
    # 定义字段与FIELD之间读配对
    fields = [('text', TEXT), ('label', LABEL)]
    
    if is_train:
        # 读取数据
        train_data, val_data = TabularDataset.splits(path=bc.data_path, train='train.csv',
                                                     validation='val.csv',
                                                     format='csv',
                                                     fields=fields,
                                                     skip_header=True)
        #  构建从本地加载的词向量
        vectors = Vectors(name=bc.embedding_loc)
        # 词表和标签映射表
        TEXT.build_vocab(train_data, val_data, vectors=vectors)
        LABEL.build_vocab(train_data, val_data)
        #创建训练集和验证集的批次迭代器
        train_iter = BucketIterator(train_data,
                                    batch_size=bc.batch_size,
                                    sort_key=lambda x: len(x.text),
                                    sort_within_batch=True,
                                    shuffle=True)

        val_iter = BucketIterator(val_data,
                                  batch_size=bc.batch_size,
                                  sort_key=lambda x: len(x.text),
                                  sort_within_batch=True,
                                  shuffle=True)
        #保存词表和标签映射表
        with open(bc.text_vocab_path, 'wb')as f:
            pickle.dump(TEXT.vocab, f)
        with open(bc.label_vocab_path, 'wb')as f:
            pickle.dump(LABEL.vocab, f)
        vocab_size = TEXT.vocab.vectors.shape
        return TEXT,LABEL,train_iter,val_iter,vocab_size
    else:
        # 加载词典
        with open(bc.text_vocab_path, 'rb')as f:
            TEXT.vocab = pickle.load(f)#词典配置到字段
        #加载标签词表
        with open(bc.label_vocab_path, 'rb')as f:
            LABEL.vocab = pickle.load(f)#配置
        vocab_size = TEXT.vocab.vectors.shape#得到词表形状
        return TEXT,LABEL,vocab_size

def transform_data(record, TEXT, LABEL):
    if not isinstance(record, dict):
        raise ValueError('Make sure data is dict')
    tokens = my_cut(record['text'])#文本清洗和分词
    res = []
    for token in tokens:#遍历每一个词，进行映射
        res.append(TEXT.vocab.stoi[token])
    data = torch.tensor(res).unsqueeze(1)
    if 'label' in list(record):#如果有标签
        #对标签进行映射
        label = torch.tensor(LABEL.vocab.stoi[record['label']])
    else:
        label = None
    return data, label 

if __name__ == '__main__':
    load_data_to_csv()
        
```

### Model (Birnn.py)
The model we used in this project is BiLstm model. The forward LSTM is combined with the backward LSTM to form BiLstm.
```
import torch
from torch import nn
from configs import BasicConfigs
bc = BasicConfigs()

class BiRNN(nn.Module):
    def __init__(self,TEXT,vocab_size, num_hiddens=bc.num_hiddens, num_layers=bc.num_layers):
        super(BiRNN, self).__init__()
        #嵌入层
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])
        #加载词向量权重
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=vocab_size[1],
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)
        
    def forward(self, inputs):
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs    

```

### Train (main.py)
```
import argparse
from torch import optim
from torch import nn
import model
from configs import BasicConfigs
from process_data import prepare_vocab
import time
import torch
import numpy as  np
bc = BasicConfigs()

# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument('--compute-val', default=True, action='store_true', help='compute validation accuracy or not, default:None')
parser.add_argument('--epoches', default=10, type=int, help='num of epoches for trainning loop, default:20')
parser.add_argument('--model-name', default='birnn', help='choose one model name for trainng')
args = parser.parse_args()

def train(net, optimizer, loss_func, train_iter, val_iter, compute_val,device, epoches, load_model_dir, save_model_dir):
    print(f'>>>We are gonna tranning {net.__class__.__name__} with epoches of {epoches}<<<')
    net = net.to(device)#将模型网络放入device
    if load_model_dir:#如果给定加载模型路径，则加载模型
        net.load_state_dict(torch.load(load_model_dir))
    batch_count = 0
    for epoch in range(epoches):#遍历epochs
        print(f'=>we are training epoch[{epoch+1}]...<=')
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for iter_num, batch in enumerate(train_iter):#遍历数据集数据
            X,y = batch.text.to(device), batch.label.squeeze(0).to(device)#获取一批数据
            score = net(X)#模型计算输出值
            loss = loss_func(score, y)#计算损失
            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播求导
            optimizer.step()#优化参数
            train_l_sum += loss.cpu().item()#得到损失的值
            train_acc_sum += (score.argmax(dim=1) == y).sum().cpu().item()#计算准确率
            n += y.shape[0]
            batch_count += 1
            train_acc = train_acc_sum / n
            if (iter_num+1) % 10 == 0:#每10步输出结果
                print("Train accuracy now is %.1f" % (round(train_acc, 3)*100)+'%')
        if compute_val:
            net.eval()
            val_acc=[]
            for iter_num, batch in enumerate(val_iter):#遍历验证集数据
                val_X = batch.text.to(device)
                val_y = batch.label.squeeze(0).to(device)
                val_score = net(val_X)#模型输出
                val_acc.append((val_score.argmax(dim=1) == val_y).sum().cpu().item() / len(val_y))#计算acc
            print("Val accuracy  is %.1f " % (round(np.mean(val_acc), 3) * 100) + '%')#输出成绩
            net.train()
        print('*' * 25)
        if (epoch+1) % 5 ==0 and save_model_dir:#每5轮保存一次模型
            print(f'saving model into => {save_model_dir}')
            torch.save(net.state_dict(), save_model_dir)
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc, time.time() - start))
              
#构建词表、数据迭代器
TEXT,_,train_iter,val_iter,vocab_size=prepare_vocab()
# 获取模型名称
net = getattr(model, args.model_name)(TEXT=TEXT,vocab_size=vocab_size)
device = bc.device#获取设备
#构建优化器
optimizer = optim.Adam(net.parameters(), lr=bc.lr,weight_decay=bc.alpha)
#构建损失函数
loss_func = nn.CrossEntropyLoss()
load_model_dir=None

if __name__ == '__main__':
    #调用train函数实现训练
    train(net=net, optimizer=optimizer, loss_func=loss_func,
          train_iter=train_iter, val_iter=val_iter,
          compute_val=args.compute_val, device=device, epoches=args.epoches,
          load_model_dir=load_model_dir, save_model_dir=bc.save_model_dir[args.model_name])
```

### Evaluation
```
import pandas as pd
import torch
import argparse
import model
from configs  import BasicConfigs
from process_data import transform_data,prepare_vocab
bc=BasicConfigs()#加载模型配置参数

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn', help='choose one model name for trainng')
args = parser.parse_args()

def evaluate(df):
    # 加载字典
    TEXT, LABEL, vocab_size = prepare_vocab(is_train=False)
    # 获取模型名称
    net = getattr(model, args.model_name)(TEXT=TEXT, vocab_size=vocab_size)
    #加载模型参数
    net.load_state_dict(torch.load(bc.save_model_dir[args.model_name]))
    result = {'correct':0, 'wrong':0}
    df_len = df.shape[0]
    for i in range(df_len):#遍历测试数据的每一行
        record = df.loc[i, :].to_dict()
        data, label = transform_data(record, TEXT, LABEL)
        score = net(data)#模型计算输出值
        if score.argmax(dim=1) == label:#判断是否正确
            result['correct'] += 1
        else:
            result['wrong'] += 1
    print(f"Classification Accuracy of Model({model.__class__.__name__})is {result['correct']/df_len} ")
    
if __name__ == '__main__':
    #读取数据
    test_data = pd.read_csv('data/test.csv')
    #评估
    evaluate(df=test_data)
```

### Flask deploys and calls (api.py)
After model training and evaluation, we need to deploy the model online. Here, flask microservice framework is used to deploy the model.We package the whole process of text preprocessing and model prediction into an API interface, which can be called by a third party.
```
import torch
import argparse
import model
from flask import Flask, request, jsonify
from process_data import transform_data,prepare_vocab
from configs import BasicConfigs
bc = BasicConfigs()
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn', help='choose one model name for trainng')
args = parser.parse_args()

app = Flask(__name__)#实例化flask的app
app.config['JSON_AS_ASCII'] = False
#加载字典
TEXT, LABEL,vocab_size=prepare_vocab(is_train=False)
# 获取模型名称
net = getattr(model, args.model_name)(TEXT=TEXT,vocab_size=vocab_size)
#加载模型参数
net.load_state_dict(torch.load(bc.save_model_dir[args.model_name]))

@app.route('/sentiment')
def sentiemnt():
    sentence = request.args.get('sentence')#提取请求中的文本数据
    record = {'text':sentence}#封装成字典
    data, _ = transform_data(record, TEXT, LABEL)#对文本进行预处理
    idx = net(data).argmax(dim=1).item()#模型预测得到类别下标
    prediction=LABEL.vocab.itos[idx]#下标转换成标签
    
    if prediction=='pos':#判断标签
        result = '积极'
    else:
        result = '消极'
    return jsonify({'data':result, 'status_code':200})#返回响应结果
    
if __name__ == '__main__':
    app.run(debug=False)#启动app服务
```

### Test (test_api.py)
```
    import requests
    url='http://localhost:5000/sentiment'
    sentence='感觉还不错，液晶电视，配有电脑。位置很好,'
    res=requests.get(url,params={'sentence':sentence})#发送请求
    print(res.json())
```
