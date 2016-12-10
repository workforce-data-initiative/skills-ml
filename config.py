"""Load configuration from a yaml file"""
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
