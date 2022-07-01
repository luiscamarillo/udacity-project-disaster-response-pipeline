# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os

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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
#from imblearn.ensemble import BalancedRandomForestClassifier

def load_data(database_filepath):
    """ Load data from SQLite database into a pandas dataframe.
    Args:
        database_filepath: database filepath string.
    Returns:
        df: pandas dataframe with clean data imported from SQLite database"""

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesCategories', engine)

    return df


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

def build_model():
    """ Create a model by first establishing a preprocessing and training pipeline, and Grid Searching for best model.
    Returns:
        model: Grid Search CV object with model pipeline and hyperparameters grid ready for training"""

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('attr_adder', FunctionTransformer(add_extra_features, validate=False))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced")))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [250, 350],
        'clf__estimator__min_samples_split': [4, 5]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=3,
                      scoring='recall_micro',
                      return_train_score=True)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """ Prints classification report, evaluating predictions over test data.
    Args:
        model: Trained model to evaluate.
        X_test: Numpy array wit test features to predict on.
        y_test: Correct labels from test data.
        category_names: list of names per possible label to predict."""
    
    y_preds = model.best_estimator_.predict(X_test)

    print(classification_report(y_test, y_preds, target_names=category_names))


def save_model(model, model_filepath):
    """ Saves trained model as a pickle file.
    Args:
        model: Trained model to save.
        X_test: Numpy array wit test features to predict on.
        model_filepath: string path to save model in."""

    joblib.dump(model.best_estimator_, model_filepath, compress=3)


def main():
    """ Imports data, and creates, trains, evaluates, and saves classification model."""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
        
        X_train = train_set['message'].values
        y_train = train_set.drop(['id', 'message', 'original', 'genre'], axis=1)

        X_test = test_set['message'].values
        y_test = test_set.drop(['id', 'message', 'original', 'genre'], axis=1)
        category_names = y_test.columns


        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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