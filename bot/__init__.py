import logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)-8s] (%(funcName)s@%(filename)s:%(lineno)d) -> %(message)s",
    filename="flipai.log",
    filemode="a",
)
