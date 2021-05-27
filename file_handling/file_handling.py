from loguru import logger
from configparser import SafeConfigParser
import pandas as pd
from pathlib import Path


@logger.catch
def config(conf_file_name: str) -> SafeConfigParser:
    """

    Read config file and return a config object. Used to designate target directories for data and models.
    Config.ini file is located in project base directory.

    :return: A ConfigParser object.
    """
    conf = SafeConfigParser()
    try:
        # TODO Fix global root directory settings.
        conf.read(conf_file_name)
    except SafeConfigParser:
        logger.error('Config.ini file not found. Aborted.')
        quit()
    logger.success('I/O info read from file "' + conf_file_name + '".')

    return conf


@logger.catch
def read_csv(conf: SafeConfigParser, input_file_name: str) -> pd.DataFrame:
    """

    Read config.ini file. Read specified input .csv file.
    :param conf: Config Parser file object.
    :param input_file_name: Filename including suffix.
    :return: pandas dataframe.
    """

    input_file_directory = Path(conf['input_files']['input_file_directory'])
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
