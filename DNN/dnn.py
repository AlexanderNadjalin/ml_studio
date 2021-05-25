# Dense Neural Network
from file_handling import file_handling as fh
from loguru import logger

# Globals
conf_file_name = 'C:\\Python projects\\ml_studio\\config.ini'
file_name = 'ETF.csv'

if __name__ == '__main__':
    conf = fh.config(conf_file_name)
    data = fh.read_csv(conf, file_name)

