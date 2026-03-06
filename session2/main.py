import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torchvision import transforms
from torch.utils.data import random_split
from typing import Tuple
from dataset import ChineseMNISTDataset
from model import SimpleLeNet
import numpy as np


hparams = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'log_interval': 100,
    'num_epochs': 10
}


def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    """
    Define the Accuracy metric in the function below by:
      (1) obtain the maximum for each predicted element in the batch to get the
        class (it is the maximum index of the num_classes array per batch sample)
        (look at torch.argmax in the PyTorch documentation)
      (2) compare the predicted class index with the index in its corresponding
        neighbor within label_batch
      (3) sum up the number of affirmative comparisons and return the summation

    Parameters:
    -----------
    predicted_batch: torch.Tensor shape: [BATCH_SIZE, N_CLASSES]
        Batch of predictions
    label_batch: torch.Tensor shape: [BATCH_SIZE, 1]
        Batch of labels / ground truths.
    """
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def train_epoch_cnn(
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        log_interval: int,
        device,
        epoch
        ) -> Tuple[float, float]:

    # Activate the train=True flag inside the model
    network.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc


@torch.no_grad() # decorator: avoid computing gradients
def test_epoch_mlp(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        criterion,
        device
        ) -> Tuple[float, float]:

    # Dectivate the train=True flag inside the model
    network.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # TODO
        #data = data.reshape(data.shape[0], -1)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output, target).item())  # sum up batch loss

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc

def main():
    seed = 123
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)
    _ = torch.mps.manual_seed(seed)

    base = "/Users/ramonviedma/upc/aidl-2026-spring-mlops/session-2-ramon/input/"
    labels_csv = base+"chinese_mnist.csv"
    img_path = base+"data/data"
    transform = transforms.ToTensor()
    print("Loading dataset...")
    mnist_dataset = ChineseMNISTDataset(csv_file=labels_csv, img_dir=img_path, transform=transform)

    train_size = int(0.6 * len(mnist_dataset))
    validate_size = int(0.3 * len(mnist_dataset))
    test_size = int(0.1 * len(mnist_dataset))

    train_dataset, validation_dataset, test_dataset = random_split(mnist_dataset, [train_size, validate_size, test_size])

    img, label = mnist_dataset[0]
    print('Img shape: ', img.shape)
    print('Label: ', label)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    network = SimpleLeNet()

    network.to(device)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=hparams['learning_rate'])
    criterion = nn.NLLLoss(reduction='mean')


    # Init lists to save the evolution of the training & test losses/accuracy.
    train_losses_mlp = []
    test_losses_mlp = []
    train_accs_mlp = []
    test_accs_mlp = []

    # For each epoch
    for epoch in range(hparams['num_epochs']):

        # Compute & save the average training loss for the current epoch
        train_loss, train_acc = train_epoch_cnn(train_dataloader,
                                                network,
                                                optimizer,
                                                criterion,
                                                hparams["log_interval"],
                                                device,
                                                epoch)
        train_losses_mlp.append(train_loss)
        train_accs_mlp.append(train_acc)

        # TODO: Compute & save the average test loss & accuracy for the current epoch
        # HELP: Review the functions previously defined to implement the train/test epochs
        test_loss, test_accuracy = test_epoch_mlp(validation_dataloader,
                                                  network,
                                                  criterion,
                                                  device)
        test_losses_mlp.append(test_loss)
        test_accs_mlp.append(test_accuracy)

    # Plot the plots of the learning curves
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('NLLLoss')
    plt.plot(train_losses_mlp, label='train')
    plt.plot(test_losses_mlp, label='validation')
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.plot(train_accs_mlp, label='train')
    plt.plot(test_accs_mlp, label='validation')
    plt.show()


if __name__ == "__main__":
    main()
