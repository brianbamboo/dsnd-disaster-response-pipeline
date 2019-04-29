import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    @param messages_filepath filepath of messages .csv file
    @param categories_filepath filepath of categories .csv file
    @return df loaded and merged dataframe
    
    Given filepaths to CSVs for the messages and categories .csv
    files, this function loads and merges the data, returning
    the merged dataframe.
    """
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    df = messages.merge(categories, on="id", how="outer")
    return df

def clean_data(df):
    """
    @param df loaded and merged messages and categories data
      from load_data() function
    @return df cleaned dataframe
    
    This function is intended to clean up the merged messages
    and categories data; in particular it handles the 'category'
    column by splitting it into 36 columns. The original column
    and duplicate rows from the entire dataset are dropped.
    """
    
    # Create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # Convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: pd.to_numeric(x))
    
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    
    # Recode data where related = 2
    df.loc[df["related"]==2, "related"] = 1
    return df

def save_data(df, database_filename):
    """
    @param df loaded, merged and cleaned dataset to save to database
    @param database_filename filename of database
    
    This function stores the data in a SQLite database, with name specified
    by parameter database_filename. The table name is DisasterResponseDataTable.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponseDataTable', engine, index=False, if_exists="replace")
    return

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