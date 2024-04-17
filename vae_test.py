import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.FC_mean(h_)
        return h_


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        product = self.FC_output(h)
        return product


class Model(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(Model, self).__init__()
        self.Encoder = Encoder(
            input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.Decoder = Decoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim
        )

    def forward(self, x):
        z = self.Encoder(x)
        x_hat = self.Decoder(z)

        return x_hat


class VAEDataset(Dataset):
    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, item):
        return self.matrix[item]


def loss_function(x, x_hat):
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = nn.functional.l1_loss(x_hat, x, reduction="sum")
    # KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss


def main():
    x_dim = 128
    hidden_dim = 400
    latent_dim = 64
    lr = 1e-3
    batch_size = 100

    epochs = 30

    model = Model(x_dim, hidden_dim, latent_dim).to("cuda")
    optimizer = Adam(model.parameters(), lr=lr)

    all_desc = np.load(
        "/home/n11373598/work/descriptor-disambiguation/output/aachen/codebook_r2d2_eigenplaces2048_2048.npy"
    )

    # print(np.min(all_desc), np.max(all_desc))
    # all_desc = MinMaxScaler().fit_transform(all_desc)
    # print(np.min(all_desc), np.max(all_desc))

    bp = int(all_desc.shape[0] * 0.75)
    train_dataset = VAEDataset(all_desc[:bp])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = VAEDataset(all_desc[bp:])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        mse_loss = 0
        for batch_idx, x in enumerate(tqdm(train_loader)):
            if x.shape[0] < batch_size:
                continue
            x = x.view(batch_size, x_dim)
            x = x.to("cuda").float()

            optimizer.zero_grad()

            x_hat = model(x)
            loss = loss_function(x, x_hat)

            overall_loss += loss.item()
            mse_loss += torch.sum(torch.abs(x_hat - x)).item()

            loss.backward()
            optimizer.step()

        mse_loss_test = 0
        with torch.no_grad():
            for x in test_loader:
                if x.shape[0] < batch_size:
                    continue
                x = x.view(batch_size, x_dim)
                x = x.to("cuda").float()
                x_hat = model(x)
                mse_loss_test += torch.sum(torch.abs(x_hat - x)).item()

        print(
            "\tEpoch",
            epoch + 1,
            "complete!",
            "\tAverage Loss: ",
            overall_loss / (batch_idx * batch_size),
        )
        print(mse_loss / batch_idx)
        print(mse_loss_test / len(test_loader))

    print("Finish!!")


if __name__ == "__main__":
    main()
