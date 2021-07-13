import os
import time
import numpy as np
import pprint
import datetime
from utils import fio
import logging


class Logger():

    def __init__(self, log_file, config):
        self.log_file = log_file
        config_message = f'Hyperparameters:\n{pprint.pformat(config)}'
        self.write(config_message)

    def write(self, text):
        log_text = f'[{datetime.datetime.now()}] {text}\n'
        fio.txt_write(self.log_file, log_text)

    def print(self, text):
        print(text)
        self.write(text)
