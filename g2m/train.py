from g2m.encoder import Encoder
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from g2m.dataset import *
import polyscope as ps
import time
from torch.utils.tensorboard import SummaryWriter
dataset = ["10000_1e-3", "36d_2000_pi"]

def train_pq(dataloader: DataLoader, model: Encoder, optimizer: optim.Optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss_avg = 0.0
    for batch, (p_prime, q) in enumerate(dataloader):
        p = model(q)
        loss = nn.MSELoss()

        output = loss(p_prime.flatten(), p)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = output.item()
        loss_avg += loss
        # if batch % 10 == 0:
        #     print(f"training loss: {loss:>7f}")
    return loss_avg / len(dataloader)

def test_pq(dataloader: DataLoader, model: Encoder):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0.0
    with torch.no_grad():
        for (p_prime, q) in dataloader:
            pred = model(q)
            test_loss += nn.MSELoss()(p_prime.flatten(), pred).item()
    test_loss /= n_batches
    # print(f"Test Error: Avg loss = {test_loss:>8f}\n")
    return test_loss

        
def train_with_pq(load_from = 0, epochs = 1000):
    name = dataset[1]
    training_data = PQDataset(name = name, end = 1500)
    testing_data = PQDataset(name = name, start = 1500)

    train_dataloader = DataLoader(training_data, batch_size = 50)
    test_dataloader = DataLoader(testing_data, batch_size = 50)
    n_modes = training_data.n_modes
    n_nodes = training_data.n_nodes
    model = Encoder(n_modes, n_nodes)
    writer = SummaryWriter()
    tot_epochs = load_from
    if load_from > 0:
        model.load_state_dict(torch.load(f"data/{name}_{load_from}.pth"))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for t in range(tot_epochs, tot_epochs + epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_pq(train_dataloader, model, optimizer)
        test_loss = test_pq(test_dataloader, model)
        writer.add_scalar("Loss/train", train_loss, t)
        writer.add_scalar("Loss/test", test_loss, t)

        if t % 50 == 0:
            print(f"epoch {t}, train loss: {train_loss:>7f}, test loss: {test_loss:>7f}")

    tot_epochs += epochs
    print("Done!")
    torch.save(model.state_dict(), f"data/{name}_{tot_epochs}.pth")
    writer.flush()

if __name__ == "__main__":
    train_with_pq(load_from = 5000, epochs = 1000)