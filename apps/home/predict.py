import pickle
import joblib
import numpy as np
import pandas as pd
import os
from core import settings
# from .utils import TrainPreprocess
from prml_helper.utils import TrainPreprocess

class Model:
    def __init__(self): 
        self.model_list = {'target': 'apps\home\ml_models\logisticpipe_target.pkl',
                            'threat_insult': 'apps\home\ml_models\logisticpipe_threat_insult.pkl',
        }
        # self.model_ = None
        # print(os.path.join(settings.CORE_DIR,self.model_list['target']))
        print("Loading target model")
        self.model_target=joblib.load(os.path.join(settings.CORE_DIR,self.model_list['target']))
        print("Loading threat_insult model")
        self.model_threat_insult = joblib.load(os.path.join(settings.CORE_DIR,self.model_list['threat_insult']))
        print("Loading complete")
        # print(self.model_)

    def predict(self,data):
        data = pd.DataFrame([data],columns=['comment_text'])
        print(data)
        prediction={
            'target':self.model_target.predict(data['comment_text']),
            'threat':self.model_threat_insult['threat'].predict(data['comment_text']),
            'insult':self.model_threat_insult['insult'].predict(data['comment_text']),
        }
        # self.model_target.predict(data['comment_text'])
        print(prediction)
        return prediction