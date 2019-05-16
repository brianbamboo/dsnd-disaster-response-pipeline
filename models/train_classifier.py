# import libraries
import sys
import re
import pandas as pd
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    @param database_filepath filepath to SQLite database
    @return X array/vector of message data
    @return Y 0/1 matrix of multi-response data
    @return category_names names of each category, for use in evaluation
    
    Given a database filepath, reads SQLite database into a
    pandas DataFrame and returns raw messsage data, response matrix
    and category names.
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table("DisasterResponseDataTable", conn)
    X = df.loc[:, "message"]
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    @param text string to text to process
    @return text list (str) of tokenized text
    
    Given a string of text, this function performs
    punctuation removal, word tokenization using NLTK,
    stop word removal and lemmatization using NLTK. The result
    is returned as a list of strings.
    """
    # Normalize case and remove punctuation
    original = text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    text = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]

    return text

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

def build_model():
    """
    @return cv GridSearchCV object
    
    Thie function defines the ML pipeline and a grid of 
    pipeline parameters to perform a grid search cross validation
    process over. Initializes and returns a scikit-learn
    GridSearchCV object with the pipeline and parameter grid.
    """
    # Define pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartingVerbExtractor())
            ])),
        ('clf', MultiOutputClassifier(estimator=OneVsRestClassifier(
            RandomForestClassifier()
        )))
    ])
    # TO COMPLETE: ADD GRID SEARCH CROSS VALIDATION
    # currently implemented without cv just to test basic script functionality 
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    TODO: add docstring
    """
    Y_preds = model.predict(X_test)
    
    # Evaluate each category
    for i in range(Y_test.shape[1]):
        print("CATEGORY: {}".format(category_names[i]))
        print("-------------")
        print(classification_report(Y_test.iloc[:, i], Y_preds[:, i]))
        
    # Evaluate overall
    acc = accuracy_score(Y_test, Y_preds)
    prec = precision_score(Y_test, Y_preds, average="weighted")
    recall = recall_score(Y_test, Y_preds, average="weighted")
    f1 = f1_score(Y_test, Y_preds, average="weighted")
    print("Overall accuracy: {}".format(acc))
    print("Overall precision: {}".format(prec))
    print("Overall recall: {}".format(recall))
    print("Overall F1 score: {}".format(f1))
    return

def save_model(model, model_filepath):
    """
    @param model prediction model (i.e. object returned from 
      build_model function)
    @param model_filepath filepath to save the pickled model to
    
    Given a model (classifier) object, pickles the model
    and writes the result to model_filepath. This function
    does not return anything.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()