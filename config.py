"""Load configuration from a yaml file"""
import logging
import os
import yaml

if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
else:
    logging.warning('No config file found, using empty config')
    config = {}
