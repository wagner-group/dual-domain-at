import logging
import os
import time

import numpy as np
import yaml


def yaml_load(filepath):
    with open(filepath) as file:
        data = yaml.safe_load(file)
    return data


def txt_write(filepath, data):
    with open(filepath, 'a') as file:
        file.write(data)


def get_exp_hash():
    exp_hash = list(
        str(int(time.time())))
    exp_hash.insert(-4, '_')
    exp_hash.insert(-8, '_')
    return ''.join(exp_hash)


def setup_text_logger(name, log_dir='./', append=False, log_level=logging.DEBUG,
                      console_out=True):
    log_format = '[%(levelname)s %(asctime)s] %(message)s'
    logging.basicConfig(level=log_level,
                        format=log_format,
                        filename=f'{log_dir}/{name}.log',
                        filemode='a' if append else 'w')

    if console_out:
        console = logging.StreamHandler()
        console.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        # Add the handler to the root logger
        logging.getLogger().addHandler(console)
