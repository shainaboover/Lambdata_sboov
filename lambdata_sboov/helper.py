'''
This module contains a date_splitter function
and MySplitter class which performs a train_val_test split
on a pandas DataFrame
'''

import pandas as pd
from IPython.display import display
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# Function to split dates ("MM/DD/YYYY", etc.) into multiple columns
def date_splitter(self, date_column_name):
    '''
    This function takes column with dates, coverts to pandas datetime
    format creates separate columns for day, month, and year
    '''

    self[date_column_name] = pd.to_datetime(
                            self[date_column_name],
                            infer_datetime_format=True)
    self['Year'] = self[date_column_name].dt.year
    self['Month'] = self[date_column_name].dt.month
    self['Day'] = self[date_column_name].dt.day
    self.drop(date_column_name, axis=1, inplace=True)
    return self


# Train/validate/test split function for a dataframe
class MySplitter():
    '''
    This class performs train_val_test split on pandas dataframe
    and prints out summary statistics
    '''
    def __init__(self, dataframe):
        self.dataframe = df


    def train_val_test_split(self, features, target,
                            train_size=0.7, val_size=0.1,
                            test_size=0.2, random_state=None,
                            shuffle=True):

        '''
        This function performs 3 way split using sklearn train_test_split
        '''

        X = df[features]
        y = df[target]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
            random_state=random_state, shuffle=shuffle)

        return X_train, X_val, X_test, y_train, y_val, y_test


    def print_split_summary(self, X_train, X_val, X_test):
        '''
        This function prints summary statistics for X_train, X_val, and X_test.
        '''

        print('######################## TRAINING DATA ########################')
        print(f'X_train Shape: {X_train.shape}')
        display(X_train.describe(include='all').transpose())
        print('')

        print('######################## VALIDATION DATA ######################')
        print(f'X_val Shape: {X_val.shape}')
        display(X_val.describe(include='all').transpose())
        print('')

        print('######################## TEST DATA ############################')
        print(f'X_test Shape: {X_test.shape}')
        display(X_test.describe(include='all').transpose())
        print('')


if __name__ == '__main__':
    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']


    # Test the MySplitter Class
    #splitter = MySplitter(df=df, features=['ash', 'hue'], target='target')
    #X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_validation_test_split()
    #splitter.print_split_summary(X_train, X_val, X_test)
