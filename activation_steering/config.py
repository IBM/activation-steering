from rich.console import Console
import os
from datetime import datetime

class LogConfig:
    """
    Configuration class for logging settings.

    Attributes:
        enabled (bool): Whether logging is enabled.
        file_output (bool): Whether to output logs to a file.
        file_path (str): Path to the log file.
    """
    def __init__(self, enabled=True):
        """
        Initialize a LogConfig instance.

        Args:
            enabled (bool): Initial enabled state for logging.
        """
        self.enabled = enabled
        self.file_output = False
        self.file_path = None

class GlobalConfig:
    """
    Singleton class for global configuration settings.

    Class Attributes:
        console (Console): Rich console instance for pretty printing.
        log_configs (dict): Dictionary of LogConfig instances for different classes.
        log_directory (str): Directory for storing log files.
    """
    _instance = None
    console = Console()
    log_configs = {
        "global": LogConfig(enabled=True),
        "LeashLayer": LogConfig(enabled=False),
        "MalleableModel": LogConfig(enabled=True),
        "SteeringVector": LogConfig(enabled=True),
        "SteeringDataset": LogConfig(enabled=True)
    }
    log_directory = "activation_steering_logs"
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance.initialize_log_files()
        return cls._instance

    @classmethod
    def initialize_log_files(cls):
        """
        Initialize log files for all configured classes.
        """
        if not cls._initialized:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(cls.log_directory, exist_ok=True)
            for class_name in cls.log_configs:
                cls.log_configs[class_name].file_path = os.path.join(cls.log_directory, f"{class_name}_{timestamp}.log")
            cls._initialized = True

    @classmethod
    def set_verbose(cls, verbose: bool, class_name: str = "global"):
        """
        Set the verbose state for a specific class or globally.

        Args:
            verbose (bool): Whether to enable verbose logging.
            class_name (str): The class name to set verbose for. Defaults to "global".
        """
        if class_name in cls.log_configs:
            cls.log_configs[class_name].enabled = verbose

    @classmethod
    def is_verbose(cls, class_name: str = "global"):
        """
        Check if verbose logging is enabled for a specific class.

        Args:
            class_name (str): The class name to check. Defaults to "global".

        Returns:
            bool: True if verbose logging is enabled, False otherwise.
        """
        return cls.log_configs[class_name].enabled and cls.log_configs["global"].enabled

    @classmethod
    def set_file_output(cls, enabled: bool, class_name: str = "global"):
        """
        Set whether to output logs to a file for a specific class or globally.

        Args:
            enabled (bool): Whether to enable file output.
            class_name (str): The class name to set file output for. Defaults to "global".
        """
        cls.initialize_log_files()  # Ensure file paths are set
        if class_name in cls.log_configs:
            cls.log_configs[class_name].file_output = enabled

    @classmethod
    def should_log_to_file(cls, class_name: str):
        """
        Check if logging to a file is enabled for a specific class.

        Args:
            class_name (str): The class name to check.

        Returns:
            bool: True if file logging is enabled, False otherwise.
        """
        return cls.log_configs[class_name].file_output or cls.log_configs["global"].file_output

    @classmethod
    def get_file_path(cls, class_name: str):
        """
        Get the log file path for a specific class.

        Args:
            class_name (str): The class name to get the file path for.

        Returns:
            str: The path to the log file.
        """
        cls.initialize_log_files()  # Ensure file paths are set
        return cls.log_configs[class_name].file_path

def log(message: str, style: str = None, class_name: str = "global"):
    """
    Log a message to the console and/or file based on the current configuration.

    Args:
        message (str): The message to log.
        style (str, optional): The style to apply to the console output.
        class_name (str): The class name associated with the log message. Defaults to "global".
    """
    if GlobalConfig.is_verbose(class_name):
        GlobalConfig.console.print(message, style=style)
    
    if GlobalConfig.should_log_to_file(class_name):
        file_path = GlobalConfig.get_file_path(class_name)
        with open(file_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")