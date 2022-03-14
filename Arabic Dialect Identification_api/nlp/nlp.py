import os
import json
from datetime import datetime
from threading import Thread, current_thread, get_ident
from typing import Dict, List, Union
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.metrics import classification_report
from transformers import pipeline

from nlp.ml_model import my_clean_str,get_pipeline
from nlp.dl_model import train_dl
class Trainer_Ml():

    def __init__(self) -> None:
        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'storage')
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)
        self.__status_path = os.path.join(
            self.__storage_path, 'model_status.json')
        self.__model_path = os.path.join(
            self.__storage_path, 'model_pickle.joblib')

        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)

        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(), "classes": [], "evaluation": {}}

        if os.path.exists(self.__model_path):
            self.model = joblib.load(self.__model_path)

        else:
            self.model = None

        self._running_threads = []
        self._pipeline = None

    def _train_job(self, x_train, x_test, y_train, y_test):
        self._pipeline.fit(x_train, y_train)
        report = classification_report(
            y_test, self._pipeline.predict(x_test), output_dict=True)
        classes = self._pipeline.classes_.tolist()
        self._update_status("Model Ready", classes, report)
        joblib.dump(self._pipeline, self.__model_path, compress=9)
        self.model = self._pipeline
        self._pipeline = None
        thread_id = get_ident()
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break
        

    def train(self, dataframe):
        if len(self._running_threads):
            raise Exception("A training process is already running.")
        dataframe["Text"]=dataframe["Text"].apply(lambda Text:my_clean_str(Text))
        x=dataframe.Text
        y=dataframe.dialect
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.02, stratify=y, random_state=42)

        self._pipeline = get_pipeline()

        # update model status
        self.model = None
        self._update_status("Training")

        t = Thread(target=self._train_job, args=(
            x_train, x_test, y_train, y_test))
        self._running_threads.append(t)
        t.start()

    def predict(self, texts) :
        response = []
        if self.model:
            preds = self.model.predict(texts)
            for i, row in enumerate(preds):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['predictions'] = row
                response.append(row_pred)
        else:
            raise Exception("No Trained model was found.")
        return response

    def get_status(self) :
        return self.model_status

    def _update_status(self, status, classes, evaluation) :
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)
            
class Trainer_Dl():

    def __init__(self) -> None:
        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'storage')
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)
        self.__status_path = os.path.join(
            self.__storage_path, 'DL_model_status.json')
        self.__model_path = os.path.join(
            self.__storage_path, 'DL_model')

        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)

        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(), "classes": [], "evaluation": {}}

        if os.path.exists(self.__model_path):
            self.model = joblib.load(self.__model_path)

        else:
            self.model = None

        self._running_threads = []
        self._pipeline = None

    def _train_job(self):
        self.trainer.train()
        self.trainer.log.trainer_state()[-1]
        report = self.trainer.log.trainer_state()[-1]
        classes = list(self.label_map.keys())
        self._update_status("Model Ready", classes, report)
        inv_label_map = inv_label_map = { v:k for k, v in self.label_map.items()}
        self.traine.model.config.label2id = self.label_map
        self.traine.model.config.id2label = inv_label_map
        self.traine.save_model(self.__model_path)
        self.train_dataset.tokenizer.save_pretrained(self.__model_path)
        # initialize pipline
        self.model = pipeline("sentiment-analysis", model=self.__model_path, device=0, return_all_scores=False)
        self._pipeline = None
        thread_id = get_ident()
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break
        

    def train(self, dataframe) :
        if len(self._running_threads):
            raise Exception("A training process is already running.")
        self.trainer,self.label_map , self.train_dataset = train_dl(dataframe)

        # update model status
        self.model = None
        self._update_status("Training")
        t = Thread(target=self._train_job)
        self._running_threads.append(t)
        t.start()

    def predict(self, texts):
        response = []
        if self.model:
            probs = self.model(texts)
            for i, row in enumerate(probs):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['predictions'] = row
                response.append(row_pred)
        else:
            raise Exception("No Trained model was found.")
        return response

    def get_status(self) -> Dict:
        return self.model_status

    def _update_status(self, status: str, classes: List[str] = [], evaluation: Dict = {}) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)