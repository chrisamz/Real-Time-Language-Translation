# data_preprocessing.py

"""
Data Preprocessing Module for Real-Time Language Translation

This module contains functions for collecting, cleaning, normalizing, and preparing text data for model training and evaluation.

Techniques Used:
- Data cleaning
- Tokenization
- Normalization
- Augmentation

Libraries/Tools:
- pandas
- numpy
- nltk
- sentencepiece

"""

import os
import pandas as pd
import numpy as np
import sentencepiece as spm
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('punkt')

class DataPreprocessing:
    def __init__(self, input_lang='en', target_lang='es', vocab_size=32000):
        """
        Initialize the DataPreprocessing class.
        
        :param input_lang: str, input language code
        :param target_lang: str, target language code
        :param vocab_size: int, size of the vocabulary for tokenization
        """
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.vocab_size = vocab_size

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing null values and duplicates.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.dropna().drop_duplicates()
        return data

    def normalize_text(self, text):
        """
        Normalize text by converting to lowercase and removing punctuation.
        
        :param text: str, input text
        :return: str, normalized text
        """
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        return text

    def tokenize_text(self, text, sp):
        """
        Tokenize text using SentencePiece model.
        
        :param text: str, input text
        :param sp: SentencePieceProcessor, SentencePiece model
        :return: str, tokenized text
        """
        return ' '.join(sp.encode_as_pieces(text))

    def train_sentencepiece(self, data, model_prefix):
        """
        Train a SentencePiece model on the provided data.
        
        :param data: list, input text data
        :param model_prefix: str, prefix for the model files
        """
        with open('temp_text.txt', 'w', encoding='utf-8') as f:
            for line in data:
                f.write(f"{line}\n")
        
        spm.SentencePieceTrainer.train(input='temp_text.txt', model_prefix=model_prefix, vocab_size=self.vocab_size)
        os.remove('temp_text.txt')

    def preprocess(self, raw_data_filepath, processed_data_dir):
        """
        Execute the full preprocessing pipeline.
        
        :param raw_data_filepath: str, path to the input data file
        :param processed_data_dir: str, directory to save processed data
        :return: DataFrame, preprocessed data
        """
        # Load data
        data = self.load_data(raw_data_filepath)

        # Clean data
        data = self.clean_data(data)

        # Normalize text
        data['input_text'] = data[f'text_{self.input_lang}'].apply(self.normalize_text)
        data['target_text'] = data[f'text_{self.target_lang}'].apply(self.normalize_text)

        # Train SentencePiece models
        self.train_sentencepiece(data['input_text'].tolist(), f'{self.input_lang}_spm')
        self.train_sentencepiece(data['target_text'].tolist(), f'{self.target_lang}_spm')

        sp_input = spm.SentencePieceProcessor()
        sp_input.load(f'{self.input_lang}_spm.model')

        sp_target = spm.SentencePieceProcessor()
        sp_target.load(f'{self.target_lang}_spm.model')

        # Tokenize text
        data['input_text'] = data['input_text'].apply(lambda x: self.tokenize_text(x, sp_input))
        data['target_text'] = data['target_text'].apply(lambda x: self.tokenize_text(x, sp_target))

        # Split data
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        
        # Save processed data
        os.makedirs(processed_data_dir, exist_ok=True)
        train_data.to_csv(os.path.join(processed_data_dir, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(processed_data_dir, 'val_data.csv'), index=False)
        print(f"Processed data saved to {processed_data_dir}")

        return train_data, val_data

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/multilingual_data.csv'
    processed_data_dir = 'data/processed/'
    input_lang = 'en'
    target_lang = 'es'
    vocab_size = 32000

    preprocessing = DataPreprocessing(input_lang, target_lang, vocab_size)

    # Preprocess the data
    train_data, val_data = preprocessing.preprocess(raw_data_filepath, processed_data_dir)
    print("Data preprocessing completed and data saved.")
