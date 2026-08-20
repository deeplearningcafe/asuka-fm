import logging
import sys
import os

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.getLogger("datasets").propagate = False
logging.getLogger("huggingface_hub").propagate = False


class Logger:
    """
    A simple wrapper class to configure the global logging state.
    Importing 'logging' in other modules will automatically use this configuration
    once 'Logger.setup_logging' is called in the main script.
    """

    @staticmethod
    def setup_logging(save_dir: str, logging_name: str):
        """
        Configures the root logger to write to both a file and the console.

        Args:
            save_dir (str): The directory where results are saved.
            exp_name (str): The name of the experiment, used for the log filename.
        """
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"{logging_name}_log.txt")

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_path, mode="w")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logging.info(f"=== Logging Initialized: {log_path} ===")
