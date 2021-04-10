from model import ViewsPredictor
import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from youtube_dataset import YoutubeDataset
import numpy as np
from sklearn.model_selection import KFold

def train_model(training_data_loader, testing_data_loader, num_epochs, model, batch_size, results, k_fold):
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    training_losses = []
    validation_losses = []
    validation_losses
    for epoch in range(0,num_epochs):
        training_loss = 0
        validation_loss = 0
        for batch, (numerical_features, views) in enumerate(training_data_loader):
            optimizer.zero_grad()
            y_pred = model(numerical_features)
            loss = criterion(y_pred, views)
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
        training_losses.append(training_loss/batch_size)
        for batch, (numerical_features, views) in enumerate(testing_data_loader):
            model.eval()
            y_pred = model(numerical_features)
            loss = criterion(y_pred, views)
            validation_loss+=loss.item()
        validation_losses.append(validation_loss/batch_size)
    results[k_fold] = validation_loss/batch_size
    return results


def get_dataset(ids, dataset, batch_size):
    sampler = SubsetRandomSampler(training_ids)
    data_loader = DataLoader(dataset, batch_size=batch_size,sampler=sampler)
    return data_loader

if __name__ == '__main__':
    videos_data = '~/Datasets/youtube/USvideos.csv'
    channels_data = '~/Datasets/youtube/channels.csv'
    dataset = YoutubeDataset(videos_data, channels_data)
    model = ViewsPredictor(dataset)
    k_folds = 5
    num_epochs = 20
    batch_size = 100
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}
    for i, (training_ids, testing_ids) in enumerate(kfold.split(dataset)):
        training_set =  get_dataset(training_ids, dataset, batch_size)
        testing_set = get_dataset(testing_ids, dataset, batch_size)
        results = train_model(training_set, testing_set, num_epochs, model, batch_size, results, kfold)
        print(f"Results at kfold {i}: {results}")
