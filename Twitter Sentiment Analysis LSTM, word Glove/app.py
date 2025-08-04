import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import os
from copy import deepcopy

# Initialize NLTK
nltk.download('wordnet', quiet=True)
tokenizer = RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

# Load GloVe embeddings
words = {}
def add_to_dict(d, filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split(' ')
                try:
                    d[line[0]] = np.array(line[1:], dtype=float)
                except:
                    continue
    except FileNotFoundError:
        st.error(f"Error: {filename} not found. Please ensure the GloVe file is in the 'model/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading GloVe file: {str(e)}")
        st.stop()

# Load GloVe embeddings
try:
    add_to_dict(words, 'model/glove.6B.50d.txt')
except Exception as e:
    st.error(f"Failed to load GloVe embeddings: {str(e)}")
    st.stop()

# Load pre-trained model
try:
    model = load_model('model/best_model.keras')
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Preprocessing functions
def message_to_token_list(s):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words]
    return useful_tokens

def message_to_word_vectors(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)
    vectors = [word_dict[token] for token in processed_list_of_tokens if token in word_dict]
    return np.array(vectors, dtype=float)

def pad_X(X, desired_sequence_length=57):
    X_copy = deepcopy(X)
    for i, x in enumerate(X):
        x_seq_len = x.shape[0] if x.shape[0] > 0 else 1
        sequence_length_difference = desired_sequence_length - x_seq_len
        pad = np.zeros(shape=(sequence_length_difference, 50))
        X_copy[i] = np.concatenate([x if x.shape[0] > 0 else np.zeros((1, 50)), pad])
    return np.array(X_copy).astype(float)

def df_to_X_y(df):
    X = []
    y = []
    for i in range(len(df)):
        tweet = df['tweet'].iloc[i]
        vectors = message_to_word_vectors(tweet)
        X.append(vectors if vectors.shape[0] > 0 else np.zeros((1, 50)))
        y.append(df['label'].iloc[i])
    return X, np.array(y)

# Streamlit UI
st.title("Twitter Tweet Sentiment Analysis")
st.markdown("Enter a tweet to predict its sentiment or upload a CSV to fine-tune the model.")

# Tweet Prediction Section
st.header("Predict Tweet Sentiment")
tweet_input = st.text_area("Enter your tweet:", height=100)
if st.button("Predict"):
    if tweet_input:
        try:
            vectors = message_to_word_vectors(tweet_input)
            padded_vectors = pad_X([vectors])
            prediction = model.predict(padded_vectors, verbose=0)[0][0]
            sentiment = "Positive" if prediction <= 0.5 else "Negative"
            confidence = float(prediction)
            st.success(f"Sentiment: **{sentiment}** (Confidence: {confidence:.2%})")
        except Exception as e:
            st.error(f"Error predicting sentiment: {str(e)}")
    else:
        st.error("Please enter a tweet.")

# Fine-Tuning Section
st.header("Fine-Tune Model")
uploaded_file = st.file_uploader("Upload a CSV file with 'tweet' and 'label' columns", type=["csv"])
if st.button("Fine-Tune Model"):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'tweet' not in df.columns or 'label' not in df.columns:
                st.error("CSV must contain 'tweet' and 'label' columns.")
            else:
                # Split data
                df = df.sample(frac=1, random_state=1).reset_index(drop=True)
                split_index = int(len(df) * 0.8)
                train_df, test_df = df[:split_index], df[split_index:]

                # Preprocess data
                with st.spinner("Preprocessing data..."):
                    X_train, y_train = df_to_X_y(train_df)
                    X_train = pad_X(X_train)
                    X_test, y_test = df_to_X_y(test_df)
                    X_test = pad_X(X_test)

                # Fine-tune model
                with st.spinner("Fine-tuning model..."):
                    cp = ModelCheckpoint('model/finetuned_model.keras', save_best_only=True)
                    frequencies = pd.Series(y_train).value_counts()
                    weights = {0: frequencies.sum() / frequencies[0], 1: frequencies.sum() / frequencies[1]}
                    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.0001), metrics=['accuracy', AUC(name='AUC')])
                    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, callbacks=[cp], class_weight=weights, verbose=0)

                # Load fine-tuned model
                model = load_model('model/finetuned_model.keras')
                
                # Evaluate model
                with st.spinner("Evaluating model..."):
                    test_predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int)
                    report = classification_report(y_test, test_predictions, output_dict=True)
                    st.success("Fine-tuning complete!")
                    st.write("### Model Performance Metrics")
                    st.write(f"**Accuracy**: {report['accuracy']:.2%}")
                    st.write(f"**Positive Class (0)**")
                    st.write(f"- Precision: {report['0']['precision']:.2%}")
                    st.write(f"- Recall: {report['0']['recall']:.2%}")
                    st.write(f"- F1-Score: {report['0']['f1-score']:.2%}")
                    st.write(f"**Negative Class (1)**")
                    st.write(f"- Precision: {report['1']['precision']:.2%}")
                    st.write(f"- Recall: {report['1']['recall']:.2%}")
                    st.write(f"- F1-Score: {report['1']['f1-score']:.2%}")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    else:
        st.error("Please upload a CSV file.")
