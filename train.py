from model import ViewsPredictor
import torch
from torch.utils.data import DataLoader, random_split
from youtube_dataset import YoutubeDataset

def train_model(training_data_loader, testing_data_loader, num_epochs, model, batch_size):
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
        print({"Epoch": epoch, "training_loss": training_loss, 'validation loss': validation_loss})





if __name__ == '__main__':
    videos_data = '~/Datasets/youtube/USvideos.csv'
    channels_data = '~/Datasets/youtube/channels.csv'
    dataset = YoutubeDataset(videos_data, channels_data)
    training_size = round(0.8*len(dataset))
    testing_size = len(dataset) - training_size
    training_data, testing_data = random_split(dataset, [training_size, testing_size])
    training_data_loader = DataLoader(dataset, batch_size=100)
    testing_data_loader = DataLoader(testing_data, batch_size=100)
    model = ViewsPredictor(dataset)
    train_model(training_data_loader, testing_data_loader, 50, model, batch_size=100)
