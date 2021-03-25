import torch
from torch.utils.data import DataLoader, random_split
from youtube_dataset import YoutubeDataset

def train_model(training_data_loader, testing_data_loader, num_epochs):
    criterion = torch.nn.MSELoss
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in num_epochs:
        for batch, numerical_features in enumerate(training_data_loader):
            inputs = torch.from_numpy(numerical_features)
            y_pred = model(inputs)
            return
    return


if __name__ == '__main__':
    videos_data = '~/Datasets/youtube/USvideos.csv'
    channels_data = '~/Datasets/youtube/channels.csv'
    dataset = YoutubeDataset(videos_data, channels_data)
    training_size = round(0.8*len(dataset))
    testing_size = len(dataset) - training_size
    training_data, testing_data = random_split(dataset, [training_size, testing_size])
    training_data_loader = DataLoader(training_data, batch_size=100)
    testing_data_loader = DataLoader(testing_data, batch_size=100)
    # train_model(training_data_loader, testing_data_loader, 10)

