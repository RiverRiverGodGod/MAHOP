# rmse_error = np.sqrt(mean_squared_error(
#     test_loader.dataset.inverse_transform(np.asarray(prediction)), 
#     test_loader.dataset.inverse_transform(np.asarray(groundtruth))))
# mea_error = mean_absolute_error(
#     test_loader.dataset.inverse_transform(np.asarray(prediction)), 
#     test_loader.dataset.inverse_transform(np.asarray(groundtruth)))
# rmse_error = np.sqrt(mean_squared_error(np.array(prediction), np.array(groundtruth)) * (test_loader.dataset.scaler1.scale_**2))
# mea_error = mean_absolute_error(np.array(prediction), np.array(groundtruth)) * (test_loader.dataset.scaler1.scale_)

# class RLPnet(nn.Module):
#     def __init__(self, feature_size, predict_w, use_seaon, use_trend):
#         super().__init__()
#         self.use_season = use_seaon
#         self.use_trend = use_trend
    
#         self.feature_size = feature_size
#         self.predict_w = predict_w

#         self.near_embedding_size = 64
#         self.season_embedding_size = 48
#         self.trend_embedding_size = 32

#         self.near_lstm_embedding = nn.LSTM(
#             input_size = self.feature_size, hidden_size = self.near_embedding_size, batch_first=True
#         )
#         self.season_lstm_embedding = nn.LSTM(
#             input_size = self.feature_size, hidden_size = self.season_embedding_size, batch_first=True
#         )
#         self.trend_lstm_embedding = nn.LSTM(
#             input_size = self.feature_size, hidden_size = self.trend_embedding_size, batch_first=True
#         )
#         embedding_size = self.near_embedding_size
#         if self.use_season: embedding_size += self.season_embedding_size
#         if self.use_trend: embedding_size += self.trend_embedding_size

#         self.MLP_layer = nn.Sequential(
#             nn.Linear(embedding_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.predict_w)
#         )
    
#     def forward(self, x :torch.Tensor):
#         embedding_vec= []

#         near_seq = x[0].to(self.device).float()
#         _, (near_embedding, _) = self.near_lstm_embedding(near_seq)
#         embedding_vec.append(near_embedding.squeeze(0))

#         if self.use_season:
#             season_seq = x[1].to(self.device).float()
#             _, (season_embedding, _) = self.season_lstm_embedding(season_seq)
#             embedding_vec.append(season_embedding.squeeze(0))
        
#         if self.use_trend:
#             trend_seq = x[2].to(self.device).float()    
#             _, (trend_embedding, _) = self.trend_lstm_embedding(trend_seq)
#             embedding_vec.append(trend_embedding.squeeze(0))
    
#         embedding_vec = torch.cat(embedding_vec, 1)

#         output = self.MLP_layer(embedding_vec)

#         return output