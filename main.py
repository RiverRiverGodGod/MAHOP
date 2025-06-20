from utils.config import ArgsConfig
from engine import Engine
# from engine_contrast import Engine
from engine_arima import Engine_ARIMA
import random
import torch
import numpy as np

if __name__ == '__main__':
    fix_seed = 2000  #30 #2000  #2**7+17 #42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = ArgsConfig.get_args()
    engine = Engine(args)
    engine.train()
