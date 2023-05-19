import sys
sys.path.insert(0, 'src')
from exception import CustomException
from logger import logging
import pandas as pd
import numpy as np
import dill
import os

def save_object(file_path,obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)