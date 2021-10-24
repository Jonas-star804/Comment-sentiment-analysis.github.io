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

import torch

~class BasicConfigs():
    #数据存放参数
    neg='data/neg'#负样本目录
    pos='data/pos'#正样本目录
    data_path='data'#分割后数据存放目录
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
    num_channels = [100, 100, 100]#卷积核个数
    # bilstm配置参数
    num_hiddens = 100#lstm神经元数
    num_layers = 1#lstm层数
    save_model_dir={#模型保存路径
        'textcnn':'model_storage/model_cnn.pt',
        'birnn':'model_storage/model_rnn.pt'
    }~


