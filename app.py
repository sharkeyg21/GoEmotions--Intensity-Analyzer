import streamlit as st
import joblib


# 1. PAGE SETUP & CONFIGURATION

st.set_page_config(
    page_title="Emotion Intensity Classifier",
    page_icon="🧠",
    layout="centered")

# Mappings used to decorate the UI output
id_to_tier = {
    "annoyance": "Low Intensity", "approval": "Low Intensity", "caring": "Low Intensity", 
    "confusion": "Low Intensity", "curiosity": "Low Intensity", "realization": "Low Intensity",
    "admiration": "Medium Intensity", "amusement": "Medium Intensity", "anger": "Medium Intensity", 
    "desire": "Medium Intensity", "disappointment": "Medium Intensity", "disapproval": "Medium Intensity", 
    "disgust": "Medium Intensity", "embarrassment": "Medium Intensity", "gratitude": "Medium Intensity", 
    "optimism": "Medium Intensity", "remorse": "Medium Intensity", "sadness": "Medium Intensity", 
    "surprise": "Medium Intensity",
    "excitement": "High Intensity", "fear": "High Intensity", "grief": "High Intensity", 
    "joy": "High Intensity", "love": "High Intensity", "nervousness": "High Intensity", 
    "pride": "High Intensity", "relief": "High Intensity", "neutral": "Neutral"
}

emotion_emojis = {
    "joy": "😂", "love": "🥰", "admiration": "🤩", "amusement": "😸", "approval": "👍",
    "caring": "❤️", "excitement": "🎉", "gratitude": "🙏", "optimism": "🌅", "relief": "😮‍💨",
    "pride": "🦚", "surprise": "😲", "realization": "💡", "curiosity": "🧐", "confusion": "😕",
    "fear": "📁", "nervousness": "😰", "anger": "😡", "annoyance": "😑", "disapproval": "👎",
    "disappointment": "😞", "disgust": "🤢", "sadness": "😢", "grief": "💔", "remorse": "😳",
    "embarrassment": "😳", "desire": "🥺", "neutral": "😐"
}

# 2. CACHED DATA LOADING

@st.cache_resource
def load_model_artifacts():
    # Look directly inside the project directory, regardless of what computer runs it
    model_path = "saved_models/svm_emotion_model.pkl"
    vectorizer_path = "saved_models/tfidf_vectorizer.pkl"
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Error: Model artifacts not found. Please verify the saved_models folder contains your .pkl files.")
        return None, None

model, vectorizer = load_model_artifacts()

# 3. USER INTERFACE

st.title("🧠 Emotion Detection & Intensity Analyzer")
st.write("Type a sentence below, and our optimized SVM model will predict the underlying emotion and map its psychological intensity tier.")

# Text Input Area
user_text = st.text_area("Enter text to analyze:", placeholder="I am absolutely thrilled with how this pipeline works!")

# Analyze Button
if st.button("Analyze Emotion", type="primary"):
    if user_text.strip() == "":
        st.warning("Please enter some text before analyzing.")
    elif model is not None and vectorizer is not None:
        
        # 1. Vectorize input text
        vectorized_input = vectorizer.transform([user_text])
        
        # 2. Generate Prediction
        predicted_emotion = model.predict(vectorized_input)[0]
        
        # 3. Retrieve Tier Mapping
        assigned_tier = id_to_tier.get(predicted_emotion, "Unknown Tier")
        emoji = emotion_emojis.get(predicted_emotion, "💬")
        
        st.markdown("---")
        st.subheader("Results Analysis")
        
        # Display nicely in layout columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Emotion", value=f"{emoji} {predicted_emotion.title()}")
            
        with col2:
            st.metric(label="Intensity Classification", value=assigned_tier)
            
        # Contextual UI color cards depending on severity/intensity
        if assigned_tier == "High Intensity":
            st.error("⚠️ This expression indicates high psychological emotional intensity.")
        elif assigned_tier == "Medium Intensity":
            st.info("💡 This expression indicates balanced, moderate emotional intensity.")
        elif assigned_tier == "Low Intensity":
            st.success("🌱 This expression indicates subtle, low emotional intensity.")