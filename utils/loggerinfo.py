import logging
import os

def get_logger(file_name, loggername):
    if not os.path.exists('log'):
            os.mkdir('log')
    file_name = os.path.join('log', file_name)
    logger = logging.getLogger(loggername)  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, encoding="utf-8", mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为继续写入模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever

    formatter = logging.Formatter('%(asctime)s - %(levelname)s >>> %(message)s', datefmt='%m-%d %H:%M:%S')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger