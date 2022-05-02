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
        self.model_list = {'target': 'apps\home\ml_models\logistic_target.sav',
                            'threat': 'apps\home\ml_models\logistic_threat_model.pkl',
                            'insult': 'apps\home\ml_models\logistic_insult_model.pkl',
                            'identity': 'apps\home\ml_models\logistic_identity_model.pkl'}
        print("Load vectorizer")
        self.word_vectorizer = joblib.load(os.path.join(settings.CORE_DIR, 'apps\home\ml_models\word_vectorizer.pkl'))
        # self.model_ = None
        # print(os.path.join(settings.CORE_DIR,self.model_list['target']))
        print("Loading target model")
        self.model_target=joblib.load(os.path.join(settings.CORE_DIR,self.model_list['target']))
        print("Loading threat model")
        self.model_threat = joblib.load(os.path.join(settings.CORE_DIR,self.model_list['threat']))
        print("Loading insult model")
        self.model_insult = joblib.load(os.path.join(settings.CORE_DIR,self.model_list['insult']))
        print("Loading identity model")
        self.model_identity = joblib.load(os.path.join(settings.CORE_DIR,self.model_list['identity']))
        print("Loading complete")
        print(self.model_target)

    def predict(self,data):
        data = pd.DataFrame([data],columns=['comment_text'])
        data = TrainPreprocess().fit(data['comment_text']).transform(data['comment_text'])
        print(data)
        data = self.word_vectorizer.transform(data)
        prediction={
            'target':self.model_target.predict(data),
            'threat':self.model_threat.predict(data),
            'insult':self.model_insult.predict(data),
            'identity':self.model_identity.predict(data)
        }
        # self.model_target.predict(data['comment_text'])
        print(prediction)
        return prediction