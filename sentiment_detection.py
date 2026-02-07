import re
import joblib
import streamlit as st
from pipeline_class import SentimentPipeline


pipeline = joblib.load('sentiment_pipeline.pkl')

st.title("📝 Amazon Review Sentiment Analyzer")
st.write("---")

st.write("Enter a product review to predict its sentiment")

user_review = st.text_area(
    "Enter a product review:",
    placeholder="Type your review here...",
    height=150
)



if st.button("🔮 Predict Sentiment", use_container_width=True):
    if user_review.strip():
        # PREPROCESSING HAPPENS AUTOMATICALLY!
        # normalize_negations is applied inside pipeline.predict()
        sentiment = pipeline.predict(user_review, return_single=True)
        
        sentiment_emoji = {
            "Positive": "😊",
            "Negative": "😞",
            "Neutral": "😐"
        }
        emoji = sentiment_emoji.get(sentiment, "❓")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Sentiment", f"{emoji} {sentiment}")
        
        proba = pipeline.predict_proba(user_review)[0]
        classes = pipeline.label_encoder.classes_
        
        with col2:
            st.write("**Confidence Scores:**")
            for cls, prob in zip(classes, proba):
                st.write(f"  {cls}: **{prob:.1%}**")
    else:
        st.warning("⚠️ Please enter a review!")

