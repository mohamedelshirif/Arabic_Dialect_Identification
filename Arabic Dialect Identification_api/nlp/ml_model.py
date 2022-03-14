import re
from  collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import pickle

def my_clean_str(text):
    
    search = ['_','-','\n','\t','&quot;']

    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    text = text.replace('أ', 'ا')
    text = text.replace('آ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('أ', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], ' ')
        
    #removing mention,english letters , digits 
    text=re.sub("[^\u0621-\u064A\u0660-\u0669\s]","",text)
    
    #trim
    text = text.strip()

    return text

with open("stop_words.pickle","rb") as f:
    stop_words=pickle.load(f)


def get_pipeline():
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5),stop_words=stop_words)
    clf = LinearSVC(random_state=0)
    pipe = make_pipeline(vec,clf)
    return pipe 