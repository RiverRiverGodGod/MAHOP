from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import ArgsConfig


class Dataset_BOOLV(Dataset):
    def __init__(self, data_path, flag, size=None, features=None, usescale=True):
        self.data_path = data_path
        self.near_w, self.season_w, self.trend_w, self.predict_w = size
        typemap = {'train': 0, 'valid': 1, 'test': 2}
        assert flag in typemap.keys()
        self.data_type = typemap[flag]
        self.features = features
        self.usescale = usescale
        self.__read_data()

    def __read_data(self):
        df_raw = pd.read_csv(self.data_path, parse_dates=['时间'])  #将时间列设为索引
        # print(df_raw)
        df_data0 = df_raw[:365][self.features].reset_index(drop=True)[90:-15].reset_index(drop=True)
        # print(df_data0)
        df_data1 = df_raw[-365:][self.features].reset_index(drop=True)[90:-15].reset_index(drop=True)
        # print(df_data0.describe(), df_data1.describe())
        window = self.season_w // 2 + 30 + self.predict_w
        data_len = len(df_data1) - window + 1 # 215 - 39 + 1 = 177

        num_test = 28
        num_train = int((data_len - num_test) * 0.9) # 134
        num_valid = data_len - num_train - num_test # 15
        # num_test = 28
        # num_valid = 28
        # num_train = data_len - num_valid - num_test
        num_len = [num_train, num_valid, num_test]

        startpoint = self.season_w // 2 + 30 - 1

        Ls = [startpoint, startpoint + num_train, startpoint + num_train + num_valid]
        Rs = [startpoint + num_train, startpoint + num_train + num_valid, startpoint + num_train + num_valid + num_test]
        #valid data point: train:[34: 168] valid[168:183] test[183:211]
        #range of data:    train:[34-35+1= 0:168+4= 172], valid[168-35+1= 134: 183+4= 187], test=[183-35+1= 149:211+4= 215]
        L, R = Ls[self.data_type], Rs[self.data_type]

# 先注释掉
        # if self.data_type == 0:
        #     plt.plot(df_data1.index[Ls[0]:Rs[0]], df_data1[Ls[0]:Rs[0]])
        #     plt.plot(df_data1.index[Ls[1]:Rs[1]], df_data1[Ls[1]:Rs[1]])
        #     plt.plot(df_data1.index[Ls[2]:], df_data1[Ls[2]:])
        #     plt.plot(df_data0.index, df_data0)
        #     plt.show()

        if self.usescale:
            self.scaler0 = MinMaxScaler()
            self.scaler0.fit(df_data0.values)
            data0 = self.scaler0.transform(df_data0.values)

            self.scaler1 = MinMaxScaler()
            train_data = df_data1[: Rs[0] + 4]
            self.scaler1.fit(train_data.values)
            data1 = self.scaler1.transform(df_data1.values)
        else:
            data0 = df_data0.values
            data1 = df_data1.values

        self.data0 = data0
        self.data1 = data1
        self.range = (L, R)
        self.len = num_len[self.data_type]

    def __getitem__(self, index):
        s_point = index + self.range[0]
        near_seq = self.data1[s_point+1-self.near_w: s_point+1]
        season_seq = self.data1[s_point-30+1-self.season_w//2: s_point-30+1+self.season_w//2]
        trend_seq = self.data0[s_point+1-(self.trend_w-self.predict_w): s_point+1+self.predict_w]
        predict_seq = self.data1[s_point+1: s_point+1+self.predict_w]

        x, y = (near_seq, season_seq, trend_seq), predict_seq
        return x, y
    def __len__(self):
        return self.len

    def inverse_transform(self, data):
        return self.scaler1.inverse_transform(data)


def generate_dataloader(args, flag='train'):
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size
    near_seq_window, season_seq_window = args.near_seq_window, args.season_seq_window
    trend_seq_window, predict_seq_window = args.trend_seq_window, args.predict_seq_window
    data_set = Dataset_BOOLV(
        data_path=args.data_path,
        flag=flag,
        size=[near_seq_window, season_seq_window, trend_seq_window, predict_seq_window],
        features=args.catelist)

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers)

    return data_loader

if __name__ == '__main__':
    args = ArgsConfig.get_args()
    train_loader = generate_dataloader(args, 'train')
    for (near_seq, season_seq, trend_seq),y in train_loader:
        print(near_seq.shape)
        print(season_seq.shape)
        print(trend_seq.shape)
