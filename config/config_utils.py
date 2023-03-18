import configparser

config = configparser.ConfigParser()
config.read('settings.ini')


def get_config():
    return config
