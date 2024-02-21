from torch import nn
from torch.nn import functional as F


class FakeDetector(nn.Module):

    def __init__(self, embedding_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.input_dim = embedding_dim
        self.output_dim = output_dim
        # self.fc1 = nn.Linear(embedding_dim, 512)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 512)
        # self.fc4 = nn.Linear(512, output_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        #self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, output_dim)
        #self.fc3 = nn.Linear(512, output_dim)


    def forward(self, x):
        #x = x.reshape(x.shape[0], x.shape[-1])
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.dropout(x)
        # x = self.fc4(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        #x = x.reshape(x.shape[0], 1, 1, x.shape[-1])
        return x


class SpeakerRecognizer(nn.Module):

    def __init__(self, embedding_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        #x = x.reshape(x.shape[0], x.shape[-1])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = x.reshape(x.shape[0], 1, 1, x.shape[-1])
        return x
