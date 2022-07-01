import json
import plotly
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import re
from textblob import TextBlob

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine

def add_extra_features(X_df):
    """ Feature engineering step to create extra word/character count relate features, add sentiment analysis feature,
        and drop message column before inputing into model.
    Args:
        X_df: numpy array with messages.
    Returns:
        X.values: numpy array with new features, eliminating the original message to add to the original X features"""

    X = pd.DataFrame(X_df, columns=['message'])
    
    # Word/Character count features
    X['word_count'] = X['message'].apply(lambda x: len(str(x).split(" ")))
    X['char_count'] = X['message'].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    X['sentence_count'] = X['message'].apply(lambda x: len(str(x).split(".")))
    X['avg_word_length'] = X['char_count'] / X['word_count']
    X['avg_sentence_lenght'] = X['word_count'] / X['sentence_count']
    
    # Sentiment Analysis
    X['sentiment'] = X['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Eliminate original column
    X.drop('message', axis=1, inplace=True)
    
    return X.values


app = Flask(__name__)

def tokenize(text):
    """ Clean, tokenize, remove stop words, and lemmatize string for NLP analysis.
    Args:
        text: text string to process.
    Returns:
        tokens: list of clean and lemmatized tokens based on input string"""

    # List of english language stop words
    stop_words = stopwords.words("english")

    # Instantiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').apply(lambda x: round(x['message'].count()/len(df)*100, 2))
    #.count()['message']/len(df)*100
    genre_names = list(genre_counts.index)
    related_counts = df.groupby('related').apply(lambda x: round(x['message'].count()/len(df)*100, 2))
    related_names = ['Unrelated', 'Related']
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    corr = df.drop(['id', 'message', 'original', 'genre'], axis=1).corr()
    corr_names = list(corr.columns)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Percentage Distribution of Message Genres',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'margin': {
                    'b': 120,
                },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Percentage Distribution of Related Messages',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Type of Message"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=corr_names,
                    y=corr_names,
                    z=corr,
                    type = 'heatmap',
                    colorscale = 'Viridis'
                )
            ],

            'layout': {
                'title': 'Categories Heatmap',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':45
                },
                'margin': {
                    'b': 160,
                    'l': 160
                },

            }
        },
    ]
    
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

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()