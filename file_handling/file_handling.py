import configparser
from shutil import rmtree
import os
from glob import glob
from loguru import logger
from configparser import ConfigParser
import pandas as pd
from pathlib import Path


class Settings(object):
    def __init__(self, cfg_path):
        self.cfg = ConfigParser()
        self.cfg.read(cfg_path)

    def get_setting(self, section, my_setting):
        try:
            ret = self.cfg.get(section, my_setting)
        except configparser.NoSectionError:
            ret = '!'
        return ret


@logger.catch
def get_run_logdir(log_dir, del_old_logs=True):
    if del_old_logs:
        # Delete old logs
        pattern = os.path.join(log_dir, "run_*")

        for item in glob(pattern):
            if not os.path.isdir(item):
                continue
            rmtree(item)

    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return Path.joinpath(Path(log_dir), run_id)


@logger.catch
def read_csv(conf: Settings,
             input_file_name: str) -> pd.DataFrame:
    """

    Read config.ini file. Read specified input .csv file.
    :param conf: Config Parser file object.
    :param input_file_name: Filename including suffix.
    :return: pandas dataframe.
    """

    input_file_directory = Path(conf.get_setting('input_files', 'input_file_directory'))
    input_file_path = Path.joinpath(input_file_directory, input_file_name)

    raw_data = pd.DataFrame()

    if file_valid(input_file_path):
        try:
            raw_data = pd.read_csv(input_file_path, sep=',', parse_dates=['DATE'])
        except ValueError as e:
            logger.error('File read failed with the following exception:')
            logger.error('   ' + str(e))
            logger.info('Aborted.')
            quit()
        else:
            logger.success('Data file "' + input_file_name + '" read.')

    raw_data = raw_data.set_index(['DATE'])

    return raw_data


@logger.catch
def file_valid(file_path: Path) -> bool:
    """

    Check if file path is valid. Otherwise Abort.
    :param file_path: File Path object (directory + file name).
    :return: Boolean.
    """
    if file_path.exists():
        return True
    else:
        logger.critical('File directory or file name is incorrect. Aborted')
        quit()
