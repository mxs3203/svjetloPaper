
import pickle
import pandas as pd
import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

import wandb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch import optim, nn

from AE import Autoencoder, AutoencoderLSTM

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


for output_size_ in [3, 4,5,6,7,8,9]:
    data = pd.read_csv("ALL_DATA.csv")
    curve_columns = data.columns[1:152]
    scaler = MinMaxScaler()
    data[curve_columns] = scaler.fit_transform(data[curve_columns])

    input_size = 151  # Number of features (channels)
    hidden_size = 32  # Number of LSTM units
    num_layers = 2 # Number of LSTM layers
    batch_size = 512
    learning_rate = 0.0001
    num_epochs = 500
    output_size = output_size_
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="SvjetloPaper",
        name='AELSTM-{}-{}'.format(output_size, batch_size),

        config={
            "learning_rate": learning_rate,
            "architecture": "AE",
            "dataset": "SvjetloPaper",
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_epochs": num_epochs,
            "output_size": output_size

        }
    )
    model = AutoencoderLSTM(latent_size=5).to(device)
    X_train, X_val = train_test_split(np.array(data[curve_columns], dtype="float"), test_size=0.15)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    # Data loader
    train_data = torch.utils.data.TensorDataset(X_train_tensor.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = torch.utils.data.TensorDataset(X_val_tensor.unsqueeze(1))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with validation
    num_epochs = 80
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        val_losses = []
        for i, data in enumerate(train_loader):
            inputs, = data
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.unsqueeze(dim=1)
            loss = criterion(outputs, inputs)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')


        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_features = []
        with torch.no_grad():
            for data in val_loader:
                inputs, = data
                outputs = model(inputs)
                all_features.append(pd.DataFrame(outputs.cpu().detach().numpy()))
                outputs = outputs.unsqueeze(dim=1)
                loss = criterion(outputs, inputs)
                val_losses.append(loss.item())
            if epoch % 10 == 0:  # every 5th epoch make umap
                print("Making a UMAP...")
                umap_data = pd.concat(all_features, ignore_index=True)
                umap_data = umap_data.sample(1000)
                reducer = umap.UMAP()
                embedding = pd.DataFrame(reducer.fit_transform(umap_data))
                embedding.columns = ['V0', 'V1']
                plt.scatter(embedding['V0'], embedding['V1'])
                plt.title('UMAP Visualization Epoch-{}'.format(epoch))
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.legend()
                run.log({"UMAP": plt})

        run.log({"Train/Loss": np.mean(train_losses), "Valid/Loss": np.mean(val_losses)})
    torch.save(model.state_dict(), 'AE-{}.pth'.format(output_size))
    run.log_model('AE-{}.pth'.format(output_size), "AutoEncoderModel")
    wandb.finish()
    print("Training finished")