
import pickle
import pandas as pd
import numpy as np
import torch
import umap
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
import seaborn as sb
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from losses import SupConLoss
from sklearn.model_selection import train_test_split
from torch import optim
from LSTMModel import LSTMModel


if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


with open("derived_data/X.pck", 'rb') as f:
    X = pickle.load(f)
with open("derived_data/Y.pck", 'rb') as f:
    Y = pickle.load(f)
data = pd.read_csv("ALL_DATA.csv")
curve_columns = data.columns[1:152]

input_size = 151  # Number of features (channels)
hidden_size = 32  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
batch_size = 1
learning_rate = 0.0001
num_epochs = 500
temp = 0.5
base_temperature = 0.5
output_size = 7
# Create the LSTM autoencoder model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, device=device, contrastive=True).to(device)
# Load the trained model
model.load_state_dict(torch.load('lstm_contrsative_model_7.pth'))
model.eval()

res_df = []
month_df = []
for label in tqdm(np.unique(Y), total=5):
    print(label)
    X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(Y, dtype=torch.long).to(device)
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    total_attributions = []
    indices = np.where(Y == label)[0]
    Y_filtered = Y[indices]
    X_filtered = X[indices, :, :]

    X_filtered = torch.tensor(X_filtered, dtype=torch.float32).to(device)
    Y_filtered = torch.tensor(Y_filtered, dtype=torch.long).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_filtered, Y_filtered),
                                               batch_size=batch_size,
                                               shuffle=True)


    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader),position=0, leave=False):
        outputs = model(inputs)
        outputs = outputs.squeeze(dim=1)
        res_df.append(pd.DataFrame(outputs.detach().cpu().numpy()))
        month_df.extend(labels.detach().cpu().numpy())


res_df = pd.concat(res_df)
res_df['Month'] = month_df
res_df.to_csv("contrastive_signatures.csv")