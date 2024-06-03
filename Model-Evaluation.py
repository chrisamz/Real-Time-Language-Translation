# model_evaluation.py

"""
Model Evaluation Module for Real-Time Language Translation

This module contains functions for evaluating the performance of the trained language models for real-time translation.

Techniques Used:
- BLEU score
- ROUGE score
- METEOR score

Libraries/Tools:
- numpy
- pandas
- nltk
- sacrebleu
- rouge_score

"""

import os
import pandas as pd
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import MarianTokenizer, MarianMTModel

class ModelEvaluation:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-es'):
        """
        Initialize the ModelEvaluation class.
        
        :param model_name: str, name of the pretrained model from Hugging Face
        """
        self.model_name = model_name
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

    def translate_text(self, texts):
        """
        Translate a list of texts using the model.
        
        :param texts: list of str, texts to be translated
        :return: list of str, translated texts
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return translations

    def calculate_bleu(self, references, hypotheses):
        """
        Calculate the BLEU score for the given references and hypotheses.
        
        :param references: list of str, reference translations
        :param hypotheses: list of str, translated hypotheses
        :return: float, BLEU score
        """
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score

    def calculate_rouge(self, references, hypotheses):
        """
        Calculate the ROUGE score for the given references and hypotheses.
        
        :param references: list of str, reference translations
        :param hypotheses: list of str, translated hypotheses
        :return: dict, ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        
        avg_scores = {
            'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
            'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
            'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores])
        }
        
        return avg_scores

    def calculate_meteor(self, references, hypotheses):
        """
        Calculate the METEOR score for the given references and hypotheses.
        
        :param references: list of str, reference translations
        :param hypotheses: list of str, translated hypotheses
        :return: float, METEOR score
        """
        meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]
        return np.mean(meteor_scores)

    def evaluate(self, test_data, output_dir):
        """
        Evaluate and visualize the performance of the translation model.
        
        :param test_data: DataFrame, test data with 'input_text' and 'target_text' columns
        :param output_dir: str, directory to save the evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        references = test_data['target_text'].tolist()
        input_texts = test_data['input_text'].tolist()
        hypotheses = self.translate_text(input_texts)

        # Calculate evaluation metrics
        bleu_score = self.calculate_bleu(references, hypotheses)
        rouge_scores = self.calculate_rouge(references, hypotheses)
        meteor_score = self.calculate_meteor(references, hypotheses)

        # Print evaluation metrics
        print(f"BLEU Score: {bleu_score}")
        print(f"ROUGE Scores: {rouge_scores}")
        print(f"METEOR Score: {meteor_score}")

        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"BLEU Score: {bleu_score}\n")
            f.write(f"ROUGE Scores: {rouge_scores}\n")
            f.write(f"METEOR Score: {meteor_score}\n")
        print(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.txt')}")

if __name__ == "__main__":
    test_data_filepath = 'data/processed/val_data.csv'
    output_dir = 'results/evaluation/'

    evaluator = ModelEvaluation(model_name='Helsinki-NLP/opus-mt-en-es')

    # Load data
    test_data = evaluator.load_data(test_data_filepath)

    # Evaluate the model
    evaluator.evaluate(test_data, output_dir)
    print("Model evaluation completed and results saved.")
