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

    def forward(self, seq: torch.Tensor):
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

        self.near_embedding_size = 64         #64
        self.season_embedding_size = 64
        self.trend_embedding_size = 64
        embedding_size = self.near_embedding_size


        self.near_tower = Series_Embedding(self.feature_size, self.near_embedding_size)
        if self.use_season:
            self.season_tower = Series_Embedding(self.feature_size, self.season_embedding_size)
        if self.use_trend:
            self.trend_tower = Series_Embedding(self.feature_size, self.trend_embedding_size)

        self.all_tower1 = Series_Embedding(self.feature_size, self.trend_embedding_size)
        self.all_tower2 = Series_Embedding(self.feature_size, self.trend_embedding_size)
        self.all_tower3 = Series_Embedding(self.feature_size, self.trend_embedding_size)
        self.all_tower4 = Series_Embedding(self.feature_size, self.trend_embedding_size)
        self.all_tower5 = Series_Embedding(self.feature_size, self.trend_embedding_size)


        self.decoder1 = nn.Sequential(
            nn.Linear(embedding_size, 256),        #256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(64, 16),       #256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(16, 4),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(embedding_size, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(64, 16),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(16, 4),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(embedding_size, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(64, 16),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(16, 4),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder4 = nn.Sequential(
            nn.Linear(embedding_size, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(64, 16),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(16, 4),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )
        self.decoder5 = nn.Sequential(
            nn.Linear(embedding_size, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(64, 16),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(16, 4),  # 256 64
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, self.predict_w)
        )

        self.tocket1 = nn.Linear(4, 1)
        self.tocket2 = nn.Linear(4, 1)
        self.tocket3 = nn.Linear(4, 1)
        self.tocket4 = nn.Linear(4, 1)



    def forward(self, x: torch.Tensor):
        near_seq, season_seq, trend_seq = x
        sum_seq=torch.cat([near_seq, season_seq, trend_seq], dim=1)
        sum_embedding=self.all_tower1(sum_seq)
        sum_embedding2 = self.all_tower2(sum_seq)
        sum_embedding3 = self.all_tower3(sum_seq)
        sum_embedding4 = self.all_tower4(sum_seq)
        sum_embedding5 = self.all_tower5(sum_seq)
        # print(sum_seq.shape)
        # print(near_seq.shape)
        # print(season_seq.shape)
        # print(trend_seq.shape)
        near_embedding = self.near_tower(near_seq)
        if self.use_season:
            season_embedding = self.season_tower(season_seq)
        if self.use_trend:
            trend_embedding = self.trend_tower(trend_seq)


        # print("near_seq",near_embedding.shape)
        # print("season_seq",season_embedding.shape)
        # print("trend_seq",trend_embedding.shape)
        # print("all_seq",sum_embedding.shape)
        tower_sum = torch.stack([sum_embedding,sum_embedding2, sum_embedding3,sum_embedding4], 2)
        # print(tower_sum.shape)
        output1 = self.decoder1(self.tocket1(tower_sum).squeeze(2)+sum_embedding5)
        output2 = self.decoder2(self.tocket2(tower_sum).squeeze(2)+sum_embedding5)
        output3 = self.decoder3(self.tocket3(tower_sum).squeeze(2)+sum_embedding5)
        output4 = self.decoder4(self.tocket4(tower_sum).squeeze(2)+sum_embedding5)
        # output = self.decoder5(sum_embedding)

        # print(sum_embedding.shape)
        # print(self.tocket1(tower_sum).squeeze(2).shape)
        # output1_final=self.gate1(torch.cat([output1,output],1))
        # output2_final = self.gate1(torch.cat([output2, output], 1))
        # output3_final = self.gate1(torch.cat([output3, output], 1))
        # output4_final = self.gate1(torch.cat([output4, output], 1))
        # print(output1_final.shape)
        # print("output1: ", output1.shape)
        # print("output2: ", output2.shape)
        # print("output3: ", output3.shape)
        # print("output4: ", output4.shape)

        # return output1_final, output2_final, output3_final, output4_final
        return output1,output2,output3,output4


if __name__ == '__main__':
    model = RLPnet(1, 1, 1, 1)
    # print(model)
    near_seq = torch.randn(16, 15, 1)
    season_seq = torch.randn(16, 10, 1)
    trend_seq = torch.randn(16, 20, 1)
    x = (near_seq, season_seq, trend_seq)
    output = model(x)