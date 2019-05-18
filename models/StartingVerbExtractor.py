import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from train_classifier_helper import tokenize

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        @param text string of text to examine
        @return True/False if text starts with a verb or not
        
        Given a string of text, returns True/False if the text
        starts with a verb using nltk's pos_tag (part-of-speech
        tagger).
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            if len(pos_tags) == 0:
                return False
            
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)