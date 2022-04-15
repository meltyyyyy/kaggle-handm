import json
from logging import config, getLogger

LOG_CONF = '../configs/log.json'


def get_logger(module_name, log_file):
    with open(LOG_CONF, 'r') as f:
        log_conf = json.load(f)

    log_conf["handlers"]["fileHandler"]["filename"] = log_file
    config.dictConfig(log_conf)
    return getLogger(module_name)
