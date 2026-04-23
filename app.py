import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('nextword.h5')


reversed_word_index = {index: word for word, index in tokenizer.word_index.items()}

def predict_next_word(seed_text, max_sequence_len=5):
    text = seed_text
    for _ in range(max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)
        output_word = reversed_word_index.get(predicted_word_index, '')
        text += " " + output_word
    return text
    


# Streamlit app
st.title("Next Word Prediction")
st.write("Enter a sequence of words to predict the next word.")

input_text = st.text_input("Input Text", "The cat is on the")
if st.button("Predict"):
    next_word = predict_next_word(input_text)
    st.write(f"Predicted Next Word: {next_word}")


