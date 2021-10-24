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
        
        
```


