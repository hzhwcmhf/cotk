import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # the dir where this file is located.
CACHE_DIR = os.path.join(CURRENT_DIR, 'dataloader_cache') # a cache dir for testing
CONFIG_DIR = os.path.join(CURRENT_DIR, 'dataloader_config') # a config dir for testing
CONFIG_FILE = os.path.join(CURRENT_DIR, 'config.json') # a config dir for testing
