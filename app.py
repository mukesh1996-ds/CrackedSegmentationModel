from Cracked_Detection.logger import logging
from Cracked_Detection.exception import AppException
import sys


try:

    a = 3/"small"
except Exception as e:
    raise AppException(e,sys)




