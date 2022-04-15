import re
import spacy
import pandas as pd
import numpy as np
import pt_core_news_sm


class DocumentClassifier:
    
    features = {}
    result = {}
    nlp = None
    
    def __init__(self, model_path = 'model/model.csv'):
        
        self.model = pd.read_csv(model_path, index_col = 'class_label')
        for keyword in self.model.columns:
            self.features[keyword] = 0
        
        for class_doc in  self.model.index.tolist():
            self.result[class_doc] = 0
        self.nlp = pt_core_news_sm.load()
            
    
    def tonkenize(self, text):
        
        doc = self.nlp(text)
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in self.features.keys():
                    self.features[token.lemma_] += 1
    
    def calculate_distance(self, a, b):

        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim
    
    def run(self, dict_text):

        text = ''
        
        for keyword in self.features:
            self.features[keyword] = 0
        
        for class_doc in self.result:
            self.result[class_doc] = 0
        
        for k in dict_text.keys():
            for i in dict_text[k]:
                if i['type'] == 'text':
                    text = text + '\n' + i['text'].lower()
                elif i['type'] == 'table':
                    for cell in i['cells']:
                        text = text + '\n' + cell['text'].lower()
                        
                
        self.tonkenize(text)
        vector_features = np.array(list(self.features.values()))
        for index, row in self.model.iterrows():
            self.result[index] = self.calculate_distance(vector_features, np.array(row))
        
        return self.result, self.features