import logging
import sys
import os

def get_logger(name: str):
    """
    Centralized logging configuration for the project.
    Logs to both stdout and 'app.log'.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add more (prevents duplicate logs)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    
    # Formatter: Timestamp - Name - Level - File:Line - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # 1. Console Handler (Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (app.log)
    log_file = os.getenv("LOG_FILE_PATH", os.path.join(os.path.dirname(__file__), "app.log"))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
