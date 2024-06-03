# model_training.py

"""
Model Training Module for Real-Time Language Translation

This module contains functions for developing and training language models for real-time translation.

Techniques Used:
- Instruction-finetuning
- Transfer learning
- Sequence-to-sequence models

Libraries/Tools:
- tensorflow
- transformers
- pandas
- numpy

"""

import os
import pandas as pd
import numpy as np
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        """
        Initialize the TranslationDataset class.
        
        :param data: DataFrame, input data with 'input_text' and 'target_text' columns
        :param tokenizer: MarianTokenizer, tokenizer for the model
        :param max_length: int, maximum length of the tokenized sequences
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        source = item['input_text']
        target = item['target_text']
        
        source_encodings = self.tokenizer(source, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        target_encodings = self.tokenizer(target, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        labels = target_encodings.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encodings.input_ids.squeeze(),
            'attention_mask': source_encodings.attention_mask.squeeze(),
            'labels': labels
        }

class ModelTraining:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-es', max_length=128, batch_size=16, epochs=3, lr=5e-5):
        """
        Initialize the ModelTraining class.
        
        :param model_name: str, name of the pretrained model from Hugging Face
        :param max_length: int, maximum length of the tokenized sequences
        :param batch_size: int, batch size for training
        :param epochs: int, number of training epochs
        :param lr: float, learning rate for the optimizer
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def train(self, train_data, val_data, output_dir='models/'):
        """
        Train the translation model.
        
        :param train_data: DataFrame, training data
        :param val_data: DataFrame, validation data
        :param output_dir: str, directory to save the trained model
        """
        train_dataset = TranslationDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = TranslationDataset(val_data, self.tokenizer, self.max_length)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=self.lr,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_data_filepath = 'data/processed/train_data.csv'
    val_data_filepath = 'data/processed/val_data.csv'
    output_dir = 'models/translation_model/'

    training = ModelTraining(model_name='Helsinki-NLP/opus-mt-en-es', max_length=128, batch_size=16, epochs=3, lr=5e-5)

    # Load data
    train_data = training.load_data(train_data_filepath)
    val_data = training.load_data(val_data_filepath)

    # Train the model
    training.train(train_data, val_data, output_dir)
    print("Model training completed and model saved.")
