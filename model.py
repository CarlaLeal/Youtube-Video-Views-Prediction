import torch
from torch.autograd import  Variable

class ViewsPredictor(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.num_features = len(dataset.numerical_variables)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(self.num_features, 1)

    def forward(self, features: Variable) -> int:
        non_zero_features = self.relu(features)
        prediction = self.linear(non_zero_features)
        return prediction
