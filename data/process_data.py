# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import langdetect
import sys

def detect_language(x):
    """ Identify what language a string is
     Args:
        x: string to analyze.
    Returns:
        lang: ISO 639-1 language code for analyzed string, or 'Other' if not recognized"""
    try:
        lang = langdetect.detect(x)
    except:
        lang = 'Other'
        
    return lang

def load_data(messages_filepath, categories_filepath):
    """ Load in the messages and categories .csv files, convert to pandas dataframes.
    Args:
        messages_filepath: messages filepath string.
        categories_filepath: categories filepath string.
    Returns:
        messages: pandas dataframe with messages data
        categories: pandas dataframe with categories data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories


def clean_data(messages, categories):
    """ Merges dataframes messages and categories, then cleans and processes the data.
    Args:
        messages: messages pandas dataframe.
        categories: categories pandas dataframe.
    Returns:
        df: clean and processed pandas dataframe merging messages and categories
    """

    # create a dataframe of the 36 individual category columns
    categories.set_index('id', inplace=True)
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # Extract a list of new column names for categories.
    category_colnames = row.str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    categories.reset_index(inplace=True)

    # merge messages dataframe with the new `categories` dataframe
    df = messages.merge(categories, on='id', how='inner')

    # drop rows with 2s
    df = df[df.related != 2]

    # drop duplicates
    df.drop_duplicates(subset=['id'], inplace=True)

    # Strip whitespace from message
    df['message'] = df['message'].str.strip()

    # Apply detect_language function, create column with message's language
    df['lang'] = df['message'].apply(detect_language)

    # Only keep english language messages
    df = df[df['lang'] == 'en']

    # Drop lang column
    df.drop('lang', axis=1, inplace=True)

    return df

def save_data(df, database_filename):
    """ Save processed and clean data in a SQL Database.
    Args:
        df: processed pandas dataframe.
        database_filename: database filename string.
    """

    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql('MessagesCategories', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse')


if __name__ == '__main__':
    main()