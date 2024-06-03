# real_time_translation.py

"""
Real-Time Translation Module for Real-Time Language Translation

This module contains functions for deploying the trained language models for real-time translation.

Techniques Used:
- Model loading
- Real-time data handling
- Translation

Libraries/Tools:
- numpy
- tensorflow
- transformers
- flask

"""

import os
import numpy as np
from flask import Flask, request, jsonify
from transformers import MarianTokenizer, MarianMTModel

app = Flask(__name__)

# Load the trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, tokenizer, model):
    """
    Translate a given text using the trained model.
    
    :param text: str, text to be translated
    :param tokenizer: MarianTokenizer, tokenizer for the model
    :param model: MarianMTModel, trained translation model
    :return: str, translated text
    """
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

@app.route('/translate', methods=['POST'])
def translate():
    """
    API endpoint to translate text using the trained model.
    
    :return: JSON response with the translated text
    """
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    translated_text = translate_text(text, tokenizer, model)
    
    return jsonify({'translated_text': translated_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
