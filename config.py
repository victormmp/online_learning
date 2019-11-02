import logging


class Config:
    def __init__(self, logger_name='Config'):
        self.Logger = self._set_logger(logger_name)

    @staticmethod
    def _set_logger(name='Config'):
        LOG_LEVEL = logging.INFO
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
        logging.root.setLevel(LOG_LEVEL)
        STREAM = logging.StreamHandler()
        STREAM.setLevel(LOG_LEVEL)
        STREAM.setFormatter(formatter)
        LOG = logging.getLogger(name)
        LOG.setLevel(LOG_LEVEL)
        LOG.addHandler(STREAM)

        return LOG