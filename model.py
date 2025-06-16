import torch
from torch import nn

class Series_Embedding(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super().__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size

        self.lstm_embedding = nn.LSTM(
            input_size=self.feature_size, hidden_size=self.embedding_size, batch_first=True
        )
    
    def forward(self, seq:torch.Tensor):
        _, (seq_embedding, _) = self.lstm_embedding(seq)
        seq_embedding = seq_embedding.squeeze(0)
        return seq_embedding

# class Series_Decoding(nn.Module):
#     def __init__(self, feature_size, embedding_size):
#         super().__init__()
#         self.feature_size = feature_size
#         self.embedding_size = embedding_size
#
#         self.Conv1d_decoding1 = nn.LSTM(
#             input_size=self.feature_size, hidden_size=self.embedding_size, batch_first=True
#         )
#         self.Conv1d_decoding2 = nn.LSTMnn.LSTM(
#             input_size=self.feature_size, hidden_size=self.embedding_size, batch_first=True
#         )
#
#     def forward(self, seq: torch.Tensor):
#         _, (seq_embedding, _) = self.lstm_embedding(seq)
#         seq_embedding = seq_embedding.squeeze(0)
#         return seq_embedding

class RLPnet(nn.Module):
    def __init__(self, feature_size, predict_w, use_seaon, use_trend):
        super().__init__()
        self.use_season = use_seaon
        self.use_trend = use_trend
    
        self.feature_size = feature_size
        self.predict_w = predict_w

        self.near_embedding_size = 64
        self.season_embedding_size = 64
        self.trend_embedding_size = 64
        embedding_size=self.near_embedding_size

        self.near_tower = Series_Embedding(self.feature_size, self.near_embedding_size)
        if self.use_season:
            self.season_tower = Series_Embedding(self.feature_size, self.season_embedding_size)
        if self.use_trend:
            self.trend_tower = Series_Embedding(self.feature_size, self.trend_embedding_size)

        self.decoder1 = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder4 = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )


        self.tocket1 = nn.Linear(3,1)
        self.tocket2 = nn.Linear(3,1)
        self.tocket3 = nn.Linear(3,1)
        self.tocket4 = nn.Linear(3,1)

    def forward(self, x :torch.Tensor):
        near_seq, season_seq, trend_seq = x
        near_embedding = self.near_tower(near_seq)
        if self.use_season:
            season_embedding = self.season_tower(season_seq)
        if self.use_trend:
            trend_embedding = self.trend_tower(trend_seq)


        # print("near_seq",near_embedding.shape)
        # print("season_seq",season_embedding.shape)
        # print("trend_seq",trend_embedding.shape)

        tower_sum = torch.stack([near_embedding,season_embedding,trend_embedding],2)
        output1 = self.decoder1(self.tocket1(tower_sum).squeeze(2))
        output2 = self.decoder2(self.tocket2(tower_sum).squeeze(2))
        output3 = self.decoder3(self.tocket3(tower_sum).squeeze(2))
        output4 = self.decoder4(self.tocket4(tower_sum).squeeze(2))

        # print("output1: ", output1.shape)
        # print("output2: ", output2.shape)
        # print("output3: ", output3.shape)
        # print("output4: ", output4.shape)

        return output1,output2,output3,output4

if __name__ == '__main__':
    model= RLPnet(1, 1, 1, 1)
    # print(model)
    near_seq=torch.randn(16,15,1)
    season_seq=torch.randn(16,10,1)
    trend_seq=torch.randn(16,20,1)
    x=(near_seq, season_seq, trend_seq)
    output=model(x)
