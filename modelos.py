from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
ONLY_CPU = False
try:
    from cuml import LinearRegression as cumlLinearRegression
    from cuml.ensemble import RandomForestRegressor as cumlRandomForestRegressor
    from cuml.svm import LinearSVR as cumlLinearSVR
except ImportError:
    print("No se encuentran instalados los paquetes para usar los modelos en GPU, solo podra usarlos en CPU")
    ONLY_CPU = True

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import KFold


class NeuralNetwork(nn.Module):

    def __init__(self,feature_count,hidden_layers_shape):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(feature_count,hidden_layers_shape[0]),
            nn.ReLU()
        )
        for shape_index in range(len(hidden_layers_shape)-1):
            in_shape = hidden_layers_shape[shape_index]
            out_shape = hidden_layers_shape[shape_index+1]
            self.hidden_layers.append(nn.Linear(in_shape,out_shape))
            self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(nn.Linear(hidden_layers_shape[-1],1))
        self.hidden_layers.append(nn.ReLU())

    def forward(self,x):
        x = self.hidden_layers(x)
        return x

class CustomDataset(Dataset):

    def __init__(self,filename):
        self.total_dataset = pd.read_csv(filename)
        self.target_values = self.total_dataset[self.total_dataset.columns[1]]
        self.total_dataset = self.total_dataset.drop(columns=[self.total_dataset.columns[0],self.total_dataset.columns[1]])
        self.dataset = self.total_dataset.to_numpy()
    
    def set_features(self,features):
        self.dataset = self.total_dataset[features].to_numpy()

    def __len__(self):
        return len(self.target_values)

    def __getitem__(self,idx):
        item = self.dataset[idx]
        target = self.target_values[idx]
        return item,target

def train(model, device, loss_fn,train_loader, optimizer, epoch):
    for t in range(epoch):
        model.train()
        for batch, (data,target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data,target = data.to(torch.float32), target.to(torch.float32)
            target_pred = model(data)
            loss = loss_fn(target_pred,torch.unsqueeze(target,1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def evaluate(model,loss_fn,device,test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (data,target) in enumerate(test_loader):
            data,target = data.to(device), target.to(device)
            data,target = data.to(torch.float32), target.to(torch.float32)
            output = model(data)
            loss = loss_fn(output,torch.unsqueeze(target,1)).item()
            test_loss += loss
    test_loss /= len(test_loader.dataset)
    return test_loss

def crearModelo(nombreModelo,gpu=False):
    modelo = None
    if nombreModelo == "Linear Regression":
        if not gpu:
            modelo = LinearRegression()
        else:
            modelo = cumlLinearRegression(algorithm='svd-qr')
    elif nombreModelo == "Random Forest":
        if not gpu:
            modelo = RandomForestRegressor(random_state=3006)
        else:
            modelo = cumlRandomForestRegressor(accuracy_metric="mse",random_state=3006,n_streams=1)
    elif nombreModelo == "SVM":
        if not gpu:
            modelo = LinearSVR(random_state=3006,max_iter=10000,dual="auto")
        else:
            modelo = cumlLinearSVR(max_iter=10000,verbose=0)
    elif nombreModelo == "XGBoost":
        if not gpu:
            modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
        else:
            modelo = xgb.XGBRegressor(tree_method="gpu_hist",random_state=3006)
    elif nombreModelo == "MLP":
        modelo = MLPRegressor(hidden_layer_sizes=(100,100),random_state=3006)
    return modelo

class NeuralNetworkModel:

    def __init__(self):
        pass

if __name__ == "__main__":
    device = "cuda"
    model = NeuralNetwork(209,(200,200,100,50)).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.MSELoss()
    dataset = CustomDataset("proc_train_desc_logps.csv")
    k_folds = 5
    batch_size = 2000

    kf = KFold(n_splits=k_folds)
    cross_validation_score = 0
    for fold, (train_idx,test_idx) in enumerate(kf.split(dataset)):
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            shuffle=False
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
            shuffle=False
        )
        train(model,device,loss_fn,train_loader,optimizer,100)
        split_score = evaluate(model,nn.MSELoss(reduction="sum"),device,test_loader)
        print(split_score)
        cross_validation_score += split_score
    cross_validation_score /= k_folds
    print(cross_validation_score)
        