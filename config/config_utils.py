import configparser
import os

settings_path = os.path.join(os.path.dirname(__file__), 'settings.ini')
config = configparser.ConfigParser()
config.read(settings_path)


def get_config():
    return config
