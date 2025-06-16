argsmap_testargs = {
    'argsname': 'testargs',
    'data_path': './data/cate_data.csv',
    'catelist': ['冰箱', '空调', '洗衣机', '电视'],
    #['冰箱', '空调', '洗衣机', '电视'],

    'near_seq_window': 30,    #10
    'season_seq_window': 10,
    'trend_seq_window': 20,
    'predict_seq_window':3,
    'train_categories':0,
    'train_epochs': 500,
    'batch_size': 16,
    'learning_rate': 0.001, #0.003,    #0.001
    'weight_decay': 0,
    'patience': 1,
    'num_workers': 1,
    'mountain_height': 0.0001, #0.007,      #0.01  #0.015
    'mountain_patience': [10,20],
    'mountain_decay': 0.001
}


class ArgsConfig:
    def __init__(self, argsmap):
        for argname, argvalue in argsmap.items():
            setattr(self, argname, argvalue)

    @classmethod
    def get_args(cls, argsmap=argsmap_testargs):
        return cls(argsmap)


if __name__ == '__main__':
    a = ArgsConfig.get_args()
    print(a.batch_size)
