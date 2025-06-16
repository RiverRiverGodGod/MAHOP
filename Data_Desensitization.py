import pandas as pd
import numpy as np
df = pd.read_csv('./data/cate_data.csv')
df.iloc[:,1:] = df.iloc[:,1:].apply(lambda x: x.apply(lambda y: abs(round(y/10)*10 + np.random.randint(-100, 100))))
df.to_csv('./data/cate_data.csv', index=False)