import streamlit as st
import joblib

# Load model
model = joblib.load("sentiment_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("Flipkart Review Sentiment Analysis")

review = st.text_area("Enter Review Text")

if st.button("Predict"):

    data = tfidf.transform([review])

    result = model.predict(data)

    if result[0] == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")
