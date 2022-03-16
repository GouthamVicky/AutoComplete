import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pickle


def suggestions(seed_text):

    
    next_words = 10
    load_model=keras.models.load_model("Companymodel.h5")
    words = pickle.load(open('words.pkl','rb'))
    load_tokenizer = pickle.load(open('tokenizer.pkl','rb'))
    max_sequence_len = max([len(x) for x in words])

    for _ in range(next_words):
        token_list = load_tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted =load_model.predict(token_list)[0]
        ERROR_THRESHOLD = 0.90
        results = [[i,r] for i,r in enumerate(predicted) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        #print(results)
        if len(results)==0:break
        for word, index in load_tokenizer.word_index.items():
            if index ==results[0][0]:
                output_word = word
                break
        seed_text += " " + output_word

    print("< ----------------------------------------------------- >")
    print("Auto Completed Text :",seed_text.upper())
    print("< ----------------------------------------------------- >")
    return seed_text.upper()

suggestions("NOVEL ANALYTICS")