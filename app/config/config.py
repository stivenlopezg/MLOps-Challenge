import sys
import logging

# Logger configuration -----------------------------------------------------------------------------------------------

logger_app_name = 'Domain clasification'
logger = logging.getLogger(logger_app_name)
logger.setLevel(logging.INFO)
consoleHandle = logging.StreamHandler(sys.stdout)
consoleHandle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)

# Project variables ---------------------------------------------------------------------------------------------------

label_col = 'domain'
text_col = "item"

# Path ----------------------------------------------------------------------------------------------------------------

path_root = "app"
