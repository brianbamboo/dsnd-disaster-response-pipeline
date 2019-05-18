import re, nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

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