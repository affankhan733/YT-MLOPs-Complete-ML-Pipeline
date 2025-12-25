import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    #Tokenize the text
    text = nltk.word_tokenize(text)
    #Remove non alphanumeric words
    text = [word for word in text if word.isalnum()]
    #remove stop words and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #stem the words
    text = [ps.stem(word) for word in text]
    #join the tokens back into a single string
    return "".join(text)

def preprocess_df(df, text_column = 'Text', target_column = 'Target'):
    try:
        logger.debug('Starting preprocessing for the Dataframe')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        #remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        #apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transfotmed')
        return df
    except KeyError as e:
        logger.error('column not found:%s', e)
        raise
    except Exception as e:
        logger.debug('Error during text normalization: %s', e)
        raise

def main(text_column = 'Text', target_column = 'Target'):
    """main function to load raw data , process it and save the processed data
    """
    try:
        #fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        #Transform the data
        train_processed_data = preprocess_df(train_data,text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        #Store the data inside data/processed
        data_path = os.path.join('./data', "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "Train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "Test_processed.csv"), index = False)

        logger.debug('Processed data saved to :%s', data_path)
    except FileNotFoundError as e:
        logger.debug('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.debug('No data:%s', e)
    except Exception as e:
        logger.debug('Failed to process the data transformation process: %s', e)
        print(f"error: {e}")

if __name__ == '__main__':
    main() 