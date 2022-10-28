import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score


test_label = pd.read_csv('./y_test.txt', header=None).to_numpy()
test_image = np.load('./X_test.npy')

class CustomDataset(data.Dataset):
    def __init__(self, img, label):
        super(CustomDataset, self).__init__()
        self.images = img
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        img = np.stack([img, img, img])
        labels = self.label[index]

        return img, labels

test_dataset = CustomDataset(test_image, test_label[:,1:])
test_dataloader = DataLoader(test_dataset, batch_size=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('./model')
model.to(device)
for i, data in enumerate(tqdm(test_dataloader)):
        imgs, _ = data
        imgs = imgs.to(device)
        predict = torch.sigmoid(model(imgs.float())).detach().cpu()
        if i==0:
            prediction = predict
        else:
            prediction = torch.cat([prediction, predict], dim=0)

auroc_test_macro = roc_auc_score(test_label[:,1:], prediction, multi_class='ovr', average='macro')
auprc_test_macro = average_precision_score(test_label[:,1:], prediction, average='macro')
auroc_test_micro = roc_auc_score(test_label[:,1:], prediction, multi_class='ovr', average='micro')
auprc_test_micro = average_precision_score(test_label[:,1:], prediction, average='micro')

f = open("./20213334_model.txt", "w")
f.write(f'20213334\nmacro AUROC : {auroc_test_macro}\nmicro AUROC : {auroc_test_micro}\nmacro AUPRC : {auprc_test_macro}\nmicro AUPRC : {auprc_test_micro}')
f.close()
