import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from csv files to a dataframe

    Args:
        messages_filepath: filepath of the messages csv file
        categories_filepath: filepath of the categories csv file

    Returns:
        df: dataframe with all info loaded
    '''
    #Read files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = messages.merge(categories, how = 'inner', on = 'id')
    return df


def clean_data(df):
    '''
    Cleans the categories and drop duplicates

    Args:
        df: dataframe we want to clean

    Returns:
        df1: dataframe with cleaned data
    '''
    categories = df.categories.str.split(';', expand = True)
    #first row to extract category columns names
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df1 = df.drop('categories', axis = 1).copy()
    df1  = pd.concat([df1, categories], axis = 1) 
    #drop duplicates
    df1 = df1[~df1.duplicated()]
    df1 = df1[df1.related != 2]
    return df1

def save_data(df, database_filename, table = 'database'):
    '''
    save data into a SQLite DB

    Args:
        df: cleaned dataframe
        database_filename: filename of the DB
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table, engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()