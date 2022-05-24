# 深度学习实验：茅台股票价格预测



---

## 实验内容

选择一个机器学习相关问题，并利用深度学习尝试解决这个问题。

## 实验要求

基本掌握一种深度学习开发库的使用

能够使用深度学习库搭建自己的深度学习模型

通过训练使模型在特定问题上得到较好的效果

## 实验过程

由于股价预测涉及到时间序列数据的处理，所以选择了经典的长短时记忆网络（LSTM）

### 数据获取

利用tushare包来获得上交所茅台从上市日（20010827）到20220516所有交易日的数据

~~~python
# 利用tushare获取股票数据
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
token = '7871394b0313e30c7da48d0f9b203f507ccea509ff8cb5e6fbbcfcdc'

assert ts.__version__ != '1.2.10'
ts.set_token(token)
pro = ts.pro_api()

def wash(df):# df: DATAFRAME  output:处理后的结果
    df = df.reset_index(drop=True)
    df = df.sort_index(ascending=False)
    col_list = df.columns.tolist()
    col_list.remove('ts_code')# 交易所代码无用
    col_list.remove('trade_date')# 交易日期无用
    col_list.remove('pre_close')# 昨天收盘价格无用
    col_list.remove('change')# 涨跌额无用数据
    col_list.remove('pct_chg')# 涨跌幅无用数据
    col_list.remove('close')# 将当天闭市价格作为预测目标
    col_list.append('close')
    return df[col_list]

def get_all_years(stock_id , start_year , end_date, autype=None):
    df = pro.query('daily',ts_code=stock_id,start_date=start_year, end_date=end_date)
    tradedate = df['trade_date']
    df = wash(df)
    print('Saving DataFrame: \n',df.head(5))
    df.to_csv('{}-all-year.csv'.format(stock_id),index=False)
    return tradedate

# 600519.SH茅台上证指数
stock_id = '600519.SH'
start_date = '20010827'# 20010827挂牌上市
end_date = '20220516'
tradedate = get_all_years(stock_id,start_date,end_date)
~~~

在获得数据之后通过wash函数，将close(当天闭市时股价)作为预测值放置



### 数据处理

首先是读取数据并分割生成训练集

~~~python
'''
读取原始数据，并生成训练样本
df             : 原始数据
column         : 要处理的列
train_end      : 训练集的终点
days_before    : 用多少天的数据来预测下一天
return_all     : 是否返回所有数据，默认 True
generate_index : 是否生成 index
'''
def getData(df, column, train_end=-300, days_before=30, return_all=True, generate_index=False):
    
    series = df[column].copy()
    
    # 划分数据
    # 0 ~ train_end 的为训练数据，但实际上，最后的 n 天只是作为 label
    # 而 train 中的 label，可用于 test
    train_series, test_series = series[:train_end], series[train_end - days_before:]
    
    # 创建训练集
    train_data = pd.DataFrame()
        
    # 通过移位，创建历史 days_before 天的数据
    for i in range(days_before):
        # 当前数据的 7 天前的数据，应该取 开始到 7 天前的数据； 昨天的数据，应该为开始到昨天的数据，如：
        # [..., 1,2,3,4,5,6,7] 昨天的为 [..., 1,2,3,4,5,6]
        # 比如从 [2:-7+2]，其长度为 len - 7
        train_data['c%d' % i] = train_series.tolist()[i: -days_before + i]
            
    # 获取对应的 label
    train_data['y'] = train_series.tolist()[days_before:]
        
                
    if return_all:
        return train_data, series, df.index.tolist()
    
    return train_data

train_data, all_series, df_index = getData(df, 'high', days_before=DAYS_BEFORE, train_end=TRAIN_END)
# print(df_index)
# 获取所有原始数据
all_series = np.array(all_series.tolist())
# tradedate = np.array(tradedate.tolist())
print(tradedate)
# 绘制原始数据的图
plt.figure(figsize=(12,8))
plt.plot(tradedate, all_series, label='real-data')

# 归一化，便于训练
train_data_numpy = np.array(train_data)
train_mean = np.mean(train_data_numpy)
train_std  = np.std(train_data_numpy)
train_data_numpy = (train_data_numpy - train_mean) / train_std
train_data_tensor = torch.Tensor(train_data_numpy)

# 创建 dataloader
train_set = TrainSet(train_data_tensor)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
~~~

利用getData处理好训练集后，对数据进行归一化处理，目的是为了消除奇异样本数据导致的不良影响

![output](D:\DL_programm\learning\stock_predict\report.assets\output.png)



### LSTM层构造

```python
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        
        self.lstm = nn.LSTM(
            input_size = 1,# 输入的特征维度，由于以天为单位处理所以是1
            hidden_size = 64,# 隐藏层特征维度
            num_layers = 1,# 隐藏层层数
            batch_first = True # 此选项表明输入输出的数据格式为（batch,seq,feature）
        )
        self.out = nn.Sequential(nn.Linear(64,1))
    
    def forward(self,x):
        r_out, (h_n,h_c) = self.lstm(x,None)
        # print(r_out)
        # print(r_out.shape)
        out = self.out(r_out[:,-1,:])# 取最后一天作为输出
        return out
```

具体的实现参考了http://colah.github.io/posts/2015-08-Understanding-LSTMs/和https://zhuanlan.zhihu.com/p/79064602

### 训练

~~~python
lstm = LSTM()

if torch.cuda.is_available():
    lstm = lstm.cuda()# cuda能用就用cuda

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)  # 优化所有的参数
loss_func = nn.MSELoss()# 均方误差损失函数

for step in range(EPOCH):
    for tx, ty in train_loader:
        
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()       
        
        output = lstm(torch.unsqueeze(tx, dim=2))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad() 
        loss.backward()  # 后向传播
        optimizer.step()
    print(step, loss.cpu())
    if step % 10:
        torch.save(lstm, 'lstm.pkl')
torch.save(lstm, 'lstm.pkl')
~~~

## 测试结果分析

我们将前4650个交易日划分为训练集，后300个交易日作为测试集，将原数据和预测结果作图如下

![predict](D:\DL_programm\learning\stock_predict\report.assets\predict.png)

可以看出图像的走势以及数值是十分贴近的，说明我们预测的效果相当不错

并且通过计算二范数来看两者之间的距离得到的结果为815.69，从数值上来讲也是相当好的结果
