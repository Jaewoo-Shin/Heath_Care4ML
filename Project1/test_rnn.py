import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

X_train_rnn = np.load('./X_train_rnn.npy')
X_test_rnn = np.load('./X_test_rnn.npy')
y_train = np.load('./y_train.npy')
y_test = np.load('./y_test.npy')




embedding_size = 64
hidden_size = 64
item_num = 1090 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(item_num, embedding_size, padding_idx=0)
        self.GRU = nn.GRU(embedding_size+2, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_time = torch.tensor(x[:,:,0]).unsqueeze(2).float()
        # print(x_time.shape)
        x_item = torch.tensor(x[:,:,1])
        x_value = torch.tensor(x[:,:,2]).unsqueeze(2).float()
        # print(x_value.shape)
        # print(x_item.shape)
        emb = self.embedding(x_item.long())
        emb = emb.float()

        # print(emb.shape)
        x_concat = torch.cat([x_time,emb,x_value], dim=2)
        output, _ = self.GRU(x_concat)
        result = self.fc1(output[:,-1,:])

        result = result.squeeze(1)
        result = self.sigmoid(result)
        return result

model = Net()

model.load_state_dict(torch.load('./rnn_parameter'))
model.eval()

train_auroc = roc_auc_score(y_train, model(X_train_rnn).detach().numpy())
train_auprc = average_precision_score(y_train, model(X_train_rnn).detach().numpy())

test_auroc = roc_auc_score(y_test, model(X_test_rnn).detach().numpy())
test_auprc = average_precision_score(y_test, model(X_test_rnn).detach().numpy())

f = open("./20213334_rnn.txt", "w")
f.write(f'20213334\ntrain_acuroc : {train_auroc}\ntrain_auprc : {train_auprc}\ntest_auroc : {test_auroc}\ntest_auprc : {test_auprc}')
f.close()