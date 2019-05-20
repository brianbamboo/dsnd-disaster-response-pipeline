import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from StartingVerbExtractor import StartingVerbExtractor

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseDataTable', engine)
Y = df.iloc[:, 4:]

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    "titlefont": {"size": 14},
                    "tickfont": {"size": 14}
                }
            }
        }
    ]

    # Visualization of number of positive examples in each category
    category_names = Y.columns
    category_pos_counts = Y.sum(axis=0)
    graph_two = {}
    graph_two["data"] = [Bar(x = category_names, y = category_pos_counts.values)]
    graph_two["layout"] = {
      "title": "Number of positive examples in each category", 
      "yaxis": {
        "title": "Count"
      },
      "xaxis": {
        "title": "Category",
        "titlefont": {"size": 14},
        "tickfont": {"size": 8}
      }
    }

    # Visualization of imbalance in each category
    category_imb = np.abs(category_pos_counts / Y.shape[0] - 0.5).sort_values(ascending=False)
    graph_three = {}
    graph_three["data"] = [Bar(x = category_imb.index, y = category_imb.values)]
    graph_three["layout"] = {
      "title": "Degree of imbalance in each category",
      "yaxis": {
        "title": "Absolute difference of category proportion from 0.5"
      },
      "xaxis": {
        "title": "Category",
        "titlefont": {"size": 14},
        "tickfont": {"size": 8}
      }
    }

    graphs.extend([graph_two, graph_three])
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    "titlefont": {"size": 14},
                    "tickfont": {"size": 14}
                }
            }
        }
    ]

    # Visualization of number of positive examples in each category
    category_names = Y.columns
    category_pos_counts = Y.sum(axis=0)
    graph_two = {}
    graph_two["data"] = [Bar(x = category_names, y = category_pos_counts.values)]
    graph_two["layout"] = {
      "title": "Number of positive examples in each category", 
      "yaxis": {
        "title": "Count"
      },
      "xaxis": {
        "title": "Category",
        "titlefont": {"size": 14},
        "tickfont": {"size": 8}
      }
    }

    # Visualization of imbalance in each category
    category_imb = np.abs(category_pos_counts / Y.shape[0] - 0.5).sort_values(ascending=False)
    graph_three = {}
    graph_three["data"] = [Bar(x = category_imb.index, y = category_imb.values)]
    graph_three["layout"] = {
      "title": "Degree of imbalance in each category",
      "yaxis": {
        "title": "Absolute difference of category proportion from 0.5"
      },
      "xaxis": {
        "title": "Category",
        "titlefont": {"size": 14},
        "tickfont": {"size": 8}
      }
    }

    graphs.extend([graph_two, graph_three])
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids, graphJSON=graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
