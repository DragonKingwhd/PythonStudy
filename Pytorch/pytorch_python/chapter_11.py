"""Chapter_11 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import train_network, Flatten, weight_reset, View, set_seed
from idlmam import AttentionAvg, GeneralScore, DotScore, AdditiveAttentionScore, ApplyAttention, getMaskByFill

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

from IPython.display import display_pdf
from IPython.display import Latex

# ====== 单元 2 (代码) ======
torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 3 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 4 (代码) ======
B = 128
epochs = 10

# ====== 单元 5 (代码) ======
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re

all_data = []
resp = urlopen("https://download.pytorch.org/tutorial/data.zip")
zipfile = ZipFile(BytesIO(resp.read()))
for line in zipfile.open("data/eng-fra.txt").readlines():
    line = line.decode('utf-8').lower()# 只用小写
    line = re.sub(r"[-.!?]+", r" ", line)# 去除标点
    source_lang, target_lang = line.split("\t")[0:2]
    all_data.append( (source_lang.strip(), target_lang.strip()) ) # (英文, 法文)

# ====== 单元 6 (代码) ======
for i in range(10):
    print(all_data[i])

# ====== 单元 7 (代码) ======
short_subset = [] # 我们实际要使用的子集
MAX_LEN = 6
for (s, t) in all_data:
    if max(len(s.split(" ")), len(t.split(" "))) <= MAX_LEN:
        short_subset.append((s,t))
print("Using ", len(short_subset), "/", len(all_data))

# ====== 单元 8 (代码) ======
SOS_token = "<SOS>" # "句子起始 token"
EOS_token = "<EOS>" # "句子结束 token"
PAD_token = "_PADDING_"

word2indx = {PAD_token:0, SOS_token:1, EOS_token:2}
for s, t in short_subset:
    for sentance in (s, t):
        for word in sentance.split(" "):
            if word not in word2indx:
                word2indx[word] = len(word2indx)
print("Size of Vocab: ", len(word2indx))
# 构建反向字典，便于稍后查看输出
indx2word = {}
for word, indx in word2indx.items():
    indx2word[indx] = word

# ====== 单元 9 (代码) ======
class TranslationDataset(Dataset):
    """
    接收一个以 (x, y) 字符串元组为元素的数据集，
    将其转换为 int64 张量元组。
    这样可以方便地对 Seq2Seq 问题进行编码。

    输入和输出目标中的字符串会按空格切分。
    """

    def __init__(self, lang_pairs, word2indx):
        """
        lang_pairs: List[Tuple[String,String]]，包含 Seq2Seq 问题的源/目标对。
        word2indx: Map[String,Int]，将输入字符串中的每个词映射为唯一 ID。
        """
        self.lang_pairs = lang_pairs
        self.word2indx = word2indx

    def __len__(self):
        return len(self.lang_pairs)

    def __getitem__(self, idx):
        x, y = self.lang_pairs[idx]
        x = SOS_token + " " + x + " " + EOS_token
        y = y + " " + EOS_token

        # 转换为整数列表
        x = [self.word2indx[w] for w in x.split(" ")]
        y = [self.word2indx[w] for w in y.split(" ")]

        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)

        return x, y
bigdataset = TranslationDataset(short_subset, word2indx)

# ====== 单元 10 (代码) ======
# 希望数据集划分结果一致
print(set_seed(42))

# ====== 单元 11 (代码) ======
train_size = round(len(bigdataset)*0.9)
test_size = len(bigdataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(bigdataset, [train_size, test_size])

def pad_batch(batch):
    """
    将 batch 中的各 item 填充至 batch 内最长 item 的长度
    """
    # 我们实际上有两种不同的最大长度！输入序列的最大长度，以及
    # 输出序列的最大长度。所以要分别确定，并按精确所需的量来 pad 输入/输出
    max_x = max([i[0].size(0) for i in batch])
    max_y = max([i[1].size(0) for i in batch])

    PAD = word2indx[PAD_token]

    # 使用 F.pad 在右侧对每个张量进行填充
    X = [F.pad(i[0], (0,max_x-i[0].size(0)), value=PAD) for i in batch]
    Y = [F.pad(i[1], (0,max_y-i[1].size(0)), value=PAD) for i in batch]

    X, Y = torch.stack(X), torch.stack(Y)

    return (X, Y), Y

train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=pad_batch)
test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=pad_batch)

# ====== 单元 12 (代码) ======
class Seq2SeqAttention(nn.Module):

    def __init__(self, num_embeddings, embd_size, hidden_size, padding_idx=None, layers=1, max_decode_length=20):
        super(Seq2SeqAttention, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.embd = nn.Embedding(num_embeddings, embd_size, padding_idx=padding_idx)

        # 我们将 hidden size 设为目标长度的一半，因为 encoder 是双向的，
        # 这样会得到 2 个隐状态表示，我们将它们拼接起来，
        # 就能得到所需的尺寸！
        self.encode_layers = nn.GRU(input_size=embd_size, hidden_size=hidden_size//2,
                                       num_layers=layers, bidirectional=True)
        # decoder 是单向的，并且我们需要使用 GRUCell，
        # 以便逐步进行解码
        self.decode_layers = nn.ModuleList([nn.GRUCell(embd_size, hidden_size)] +
                                     [nn.GRUCell(hidden_size, hidden_size) for i in range(layers-1)])
        self.score_net = DotScore(hidden_size)
        # predict_word 是一个小的全连接网络，用于把注意力机制的结果
        # 与局部上下文转换成对下一个词的预测
        self.predict_word = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings)
        )
        self.max_decode_length = max_decode_length
        self.apply_attn = ApplyAttention()
    
    def forward(self, input):
        # 输入应为 (B, T) 或 ((B, T), (B, T'))
        if isinstance(input, tuple):
            input, target = input
        else:
            target = None
        # batch 大小是多少？
        B = input.size(0)
        # 输入最大时间步数是多少？
        T = input.size(1)

        x = self.embd(input) #(B, T, D)

        # 获取当前模型所在的设备
        # 稍后会用到
        device = x.device

        mask = getMaskByFill(x)

        # 利用 mask 计算
        # 每条输入序列的长度
        seq_lengths = mask.sum(dim=1).view(-1) # 形状为 (B)，包含非零值数量
        # 序列长度将用于为 encoder RNN 构造 packed 输入
        x_packed = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        h_encoded, h_last = self.encode_layers(x_packed)
        h_encoded, _ = pad_packed_sequence(h_encoded) # (B, T, 2, D//2)，因为是双向的
        h_encoded = h_encoded.view(B, T, -1) #(B, T, D)
        # 现在 h_encoded 就是 encoder RNN 在输入上运行后的结果！


        # 获取最后的隐状态稍微麻烦一些
        # 首先把输出 reshape 为 (num_layers, directions, batch_size, hidden_size)
        # 然后取第一个维度的最后一个索引，因为我们想要的是
        # 最后一层的输出
        hidden_size = h_encoded.size(2)
        h_last = h_last.view(-1, 2, B, hidden_size//2)[-1,:,:,:] # 形状现在是 (2, B, D/2)
        # 然后调整为 (B, 2, D/2)，并将最后两维展平为 (B, D)
        h_last = h_last.permute(1, 0, 2).reshape(B, -1)


        # 编码部分结束。h_encoded 现在包含了输入数据的表示！
        # h_last 是 RNN 的最终输出，将作为 decoder 的初始输入状态

        # decoder 的第一个输入是 encoder 最后一步的输出
        #decoder_input = h_last

        # decoder 的新隐状态
        h_prevs = [h_last for l in range(len(self.decode_layers))]

        # 保存所有注意力机制的结果，便于稍后可视化！
        all_attentions = []
        all_predictions = []

        # 取输入中的最后一个元素（应当是 EOS 标记）
        # 作为 decoder 的首个输入
        # 也可以直接硬编码使用 SOS 标记
        decoder_input = self.embd(input.gather(1,seq_lengths.view(-1,1)-1).flatten()) #(B, D)

        # 应该解码多少步？
        steps = min(self.max_decode_length, T)
        # 如果在训练，目标值会告诉我们
        # 确切的步数
        if target is not None: # 我们已知精确的解码长度！
            steps = target.size(1)

        # 使用 teacher forcing（true）还是 auto-regressive（false）
        teacher_forcing = np.random.choice((True,False))
        for t in range(steps):
            x_in = decoder_input #(B, D)

            for l in range(len(self.decode_layers)):
                h_prev = h_prevs[l]
                h = self.decode_layers[l](x_in, h_prev)

                h_prevs[l] = h
                x_in = h
            h_decoder = x_in # (B, D)，现在得到了 decoder 在该时间步的隐状态

            # 这就是注意力机制，让我们查看所有之前的编码状态，
            # 看看哪些是相关的

            scores = self.score_net(h_encoded, h_decoder) #(B, T, 1)
            context, weights = self.apply_attn(h_encoded, scores, mask=mask)

            # 保存注意力权重以便稍后可视化
            all_attentions.append( weights.detach() ) # 这里 detach 权重是因为
            # 不再需要用它们参与任何计算，只想保存它们的
            # 数值用于做可视化

            # 现在通过拼接注意力结果与初始上下文
            # 来计算最终的表示
            word_pred = torch.cat((context, h_decoder), dim=1) #(B, D) + (B, D)  -> (B, 2*D)
            # 然后通过一个小的全连接网络
            # 得到下一个 token 的预测
            word_pred = self.predict_word(word_pred) #(B, 2*D) -> (B, V)
            all_predictions.append(word_pred)

            # 现在得到了 $\hat{y}_t$！需要选择下一时间步的输入。
            # 这里使用 torch.no_grad()，因为梯度会通过
            # RNN 的隐状态流动，而不是输入 token
            with torch.no_grad():
                if self.training:
                    if target is not None and teacher_forcing:
                        # 我们有 target 且选择了 teacher forcing，所以使用
                        # 正确的下一个答案
                        next_words = target[:,t].squeeze()
                    else:
                        # 根据已有的预测采样下一个 token
                        next_words = torch.multinomial(F.softmax(word_pred, dim=1), 1)[:,-1]
                else:
                    # 我们要做真正的预测，所以取最可能的词
                    # 也可以像 CharRNN 模型那样使用温度和采样来改进！
                    next_words = torch.argmax(word_pred, dim=1)
            # torch.no_grad() 结束

            # 已决定下一个 token，恢复使用梯度计算，
            # 这样在训练过程中 embedding 层才会被
            # 正确地调整。
            decoder_input = self.embd(next_words.to(device))

        # 解码完成！
        if self.training: # 训练时只关心预测结果
            return torch.stack(all_predictions, dim=1)
        else:# 评估时还想查看注意力权重
            return torch.stack(all_predictions, dim=1), torch.stack(all_attentions, dim=1).squeeze()

# ====== 单元 13 (代码) ======
print(set_seed(42))

# ====== 单元 14 (代码) ======
epochs = 20
seq2seq = Seq2SeqAttention(len(word2indx), 64, 256, padding_idx=word2indx[PAD_token], layers=3, max_decode_length=MAX_LEN+2)
for p in seq2seq.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -10, 10))

# ====== 单元 15 (代码) ======
def CrossEntLossTime(x, y):
    """
    x: 形状为 (B, T, V) 的输出
    y: 形状为 (B, T') 的标签
    """
    if isinstance(x, tuple):
        x, _ = x
    # 我们不希望对已经被 padding 的位置计算损失！
    cel = nn.CrossEntropyLoss(ignore_index=word2indx[PAD_token])
    T = min(x.size(1), y.size(1))
    
    loss = 0
    for t in range(T):
        loss += cel(x[:,t,:], y[:,t])
    return loss

# ====== 单元 16 (代码) ======
seq2seq_results = train_network(seq2seq, CrossEntLossTime, train_loader,epochs=epochs, device=device)

# ====== 单元 17 (代码) ======
sns.lineplot(x='epoch', y='train loss', data=seq2seq_results, label='Seq2Seq')
plt.show()

# ====== 单元 18 (代码) ======
def plot_heatmap(src, trg, scores):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='gray')

    ax.set_xticklabels(trg, minor=False, rotation='vertical')
    ax.set_yticklabels(src, minor=False)

    # 将主要刻度放在每个单元格的中心
    # 并把 x 轴刻度放到顶部
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()

# ====== 单元 19 (代码) ======
seq2seq = seq2seq.eval().cpu()
def results(indx):
    eng_x, french_y = test_dataset[indx]
    eng_str = " ".join([indx2word[i] for i in eng_x.cpu().numpy()])
    french_str = " ".join([indx2word[i] for i in french_y.cpu().numpy()])
    print("Input:     ", eng_str)
    print("Target:    ", french_str)
    
    with torch.no_grad():
        preds, attention = seq2seq(eng_x.unsqueeze(0))
        p = torch.argmax(preds, dim=2)
    pred_str = " ".join([indx2word[i] for i in p[0,:].cpu().numpy()])
    print("Predicted: ", pred_str)
    plot_heatmap(eng_str.split(" "), pred_str.split(" "), attention.T.cpu().numpy())

# ====== 单元 20 (代码) ======
print(results(12))

# ====== 单元 21 (代码) ======
print(results(13))

# ====== 单元 22 (代码) ======
print(results(16))

