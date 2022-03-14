#importing modules
import torch
device = torch.device("cuda")

import pandas as pd
import numpy as np
import random


from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from nlp.arabert.preprocess import ArabertPreprocessor
from torch.utils.data import  Dataset

from transformers import ( AutoModelForSequenceClassification,
                          AutoTokenizer,  Trainer,
                          TrainingArguments)
from transformers.data.processors.utils import InputFeatures




#preprossing
model_name = 'aubmindlab/bert-base-arabertv02-twitter'
arabic_prep = ArabertPreprocessor(model_name,keep_emojis=False)



#intiallizing the tokenizer
tok = AutoTokenizer.from_pretrained(model_name)
max_len = 100
# create function to load the data
class ClassificationDataset(Dataset):
    def __init__(self, text, target, model_name, max_len, label_map):
        super(ClassificationDataset).__init__()
        """
        Args:
        text (List[str]): List of the training text
        target (List[str]): List of the training labels
        tokenizer_name (str): The tokenizer name (same as model_name).
        max_len (int): Maximum sentence length
        label_map (Dict[str,int]): A dictionary that maps the class labels to integer
        """
        self.text = text
        self.target = target
        self.tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.label_map = label_map
      

    def __len__(self):
        return len(self.text)

    def __getitem__(self,item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer(
          text,
          max_length=self.max_len,
          padding='max_length',
          truncation=True
        )      
        return InputFeatures(**inputs,label=self.label_map[self.target[item]])


def compute_metrics(p): #p should be of type EvalPrediction
    preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(p.label_ids)
    macro_f1 = f1_score(p.label_ids,preds,average='macro')
    acc = accuracy_score(p.label_ids,preds)
    return {       
    'macro_f1' : macro_f1,
    'accuracy': acc}
    

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
training_args = TrainingArguments( 
    output_dir= "./train_log",    
    adam_epsilon = 1e-8,
    learning_rate = 2e-5,
    fp16 = True, # enable this when using V100 or T4 GPU
    per_device_train_batch_size = 16, # up to 64 on 16GB with max len of 128
    per_device_eval_batch_size = 128,
    gradient_accumulation_steps = 2, # use this to scale batch size without needing more memory
    num_train_epochs= 2,
    warmup_ratio = 0,
    do_eval = True,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True, # this allows to automatically get the best model at the end based on whatever metric we want
    metric_for_best_model = 'accuracy',
    greater_is_better = True,
    seed = 42
  )

set_seed(training_args.seed)
def train_dl(dataframe):
    #getting data
    df_train, df_test = train_test_split(dataframe, test_size=.2, random_state=42)
    df_train["Text"] = df_train["Text"].apply(lambda x: arabic_prep.preprocess(x))
    df_test["Text"] = df_test["Text"].apply(lambda x: arabic_prep.preprocess(x))  
    label_map = { v:index for index, v in enumerate(dataframe.dialect.unique()) }
    train_dataset = ClassificationDataset(df_train["Text"].to_list(), df_train["dialect"].to_list(),  model_name, max_len,label_map)
    test_dataset = ClassificationDataset(df_test["Text"].to_list(),df_test["dialect"].to_list(),model_name,max_len,label_map)
    # Create a function that return a pretrained model ready to do classification
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=len(label_map))
    trainer = Trainer(
        model = model_init(),
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics)
    return trainer , label_map,train_dataset
