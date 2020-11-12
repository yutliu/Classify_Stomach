import os
import logging
import time
# log.Log_Info('nihaohaohao')  
# 设置log的存储文件  
# os.makedirs("./logs")
# logging.basicConfig(filename = os.path.join('./logs/report_log.txt'), level=logging.DEBUG)
# logging.info('')

def creat_logger():
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    filename = time.strftime("%Y%m%d_%H%M", time.localtime())
    fhlr = logging.FileHandler(f'logs/{filename}.log') # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info('This is Logs')
    # logger.debug('this is debug')
    return logger