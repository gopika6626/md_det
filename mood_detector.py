
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.express as px
import re
import requests
import random

# Load pre-trained emotion detection model
@st.cache_resource
def load_model():
    # Using a fine-tuned model for emotion detection
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return emotion_pipeline

def clean_text(text):
    """Basic text cleaning"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def detect_subtle_emotions(text):
    """Detect subtle emotions like irritation, frustration, dismissiveness"""
    text_lower = text.lower()
    
    # Patterns for irritation/frustration
    irritation_patterns = [
        'whatever', 'fine', 'not listening', 'fed up', 'sick of', 'tired of',
        'done with', 'over it', 'seriously', 'really?', 'come on',
        'annoying', 'irritating', 'bothering', 'getting on my nerves'
    ]
    
    # Dismissive patterns  
    dismissive_patterns = [
        'do whatever you want', 'i don\'t care', 'forget it', 'never mind',
        'not my problem', 'not worth it', 'pointless'
    ]
    
    # Passive aggressive patterns
    passive_aggressive_patterns = [
        'okay sure', 'if you say so', 'good for you', 'that\'s nice',
        'how wonderful', 'great job'
    ]
    
    irritation_score = sum(1 for pattern in irritation_patterns if pattern in text_lower)
    dismissive_score = sum(1 for pattern in dismissive_patterns if pattern in text_lower)
    passive_agg_score = sum(1 for pattern in passive_aggressive_patterns if pattern in text_lower)
    
    if irritation_score > 0:
        return 'Irritation', min(70 + irritation_score * 10, 95)
    elif dismissive_score > 0:
        return 'Dismissive', min(65 + dismissive_score * 15, 95)
    elif passive_agg_score > 0:
        return 'Passive-Aggressive', min(60 + passive_agg_score * 15, 90)
    
    return None, 0

def analyze_mood(text, emotion_pipeline):
    """Analyze mood from text"""
    if not text.strip():
        return None
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # First check for subtle emotions
    subtle_emotion, subtle_confidence = detect_subtle_emotions(text)
    
    # Get emotion predictions from AI model
    emotions = emotion_pipeline(cleaned_text)[0]
    
    # Convert to more readable format
    emotion_scores = {emotion['label'].title(): round(emotion['score'] * 100, 2) 
                     for emotion in emotions}
    
    # Get dominant emotion from AI
    ai_dominant = max(emotion_scores, key=emotion_scores.get)
    ai_confidence = emotion_scores[ai_dominant]
    
    # Use subtle emotion if detected and confidence is reasonable
    if subtle_emotion and subtle_confidence > ai_confidence:
        dominant_emotion = subtle_emotion
        confidence = subtle_confidence
        # Add subtle emotion to scores
        emotion_scores[subtle_emotion] = subtle_confidence
    else:
        dominant_emotion = ai_dominant
        confidence = ai_confidence
    
    return {
        'dominant_emotion': dominant_emotion,
        'confidence': confidence,
        'all_emotions': emotion_scores
    }

def get_mood_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'Joy': '😊',
        'Sadness': '😢',
        'Anger': '😠',
        'Fear': '😨',
        'Surprise': '😲',
        'Disgust': '🤢',
        'Love': '💕',
        'Irritation': '😤',
        'Dismissive': '🙄',
        'Passive-Aggressive': '😒'
    }
    return emoji_map.get(emotion, '😐')

def get_gif_for_emotion(emotion):
    """Get a relevant GIF for the detected emotion"""
    # Free Giphy API key (public beta key - you should get your own for production)
    # Get free API key at: https://developers.giphy.com/
    GIPHY_API_KEY = "GlVGYHkr3WSBnllca54iNt0yFbjz7L65"  # Public beta key
    
    # Map emotions to search terms
    gif_search_terms = {
        'Joy': ['happy', 'celebration', 'excited', 'dancing'],
        'Sadness': ['sad', 'crying', 'disappointed', 'heartbroken'],
        'Anger': ['angry', 'mad', 'frustrated', 'rage'],
        'Fear': ['scared', 'anxious', 'worried', 'nervous'],
        'Surprise': ['surprised', 'shocked', 'wow', 'amazed'],
        'Disgust': ['disgusted', 'gross', 'yuck', 'eww'],
        'Love': ['love', 'heart', 'romantic', 'affection'],
        'Irritation': ['annoyed', 'irritated', 'eye roll', 'frustrated'],
        'Dismissive': ['whatever', 'eye roll', 'dismissive', 'done'],
        'Passive-Aggressive': ['sarcastic', 'fake smile', 'passive aggressive', 'eye roll']
    }
    
    # Get search terms for the emotion
    search_terms = gif_search_terms.get(emotion, ['neutral', 'ok'])
    search_term = random.choice(search_terms)
    
    try:
        # Make API request to Giphy
        url = f"https://api.giphy.com/v1/gifs/search"
        params = {
            'api_key': GIPHY_API_KEY,
            'q': search_term,
            'limit': 10,
            'rating': 'g',  # Keep it family-friendly
            'lang': 'en'
        }
        
        response = requests.get(url, params=params, timeout=3)
        data = response.json()
        
        if data['data']:
            # Pick a random GIF from the results
            gif = random.choice(data['data'])
            return gif['images']['fixed_height']['url']
        else:
            return None
            
    except Exception as e:
        # If API fails, return None
        return None

def get_fallback_gif(emotion):
    """Fallback GIFs if API fails"""
    fallback_gifs = {
        'Joy': 'https://media.giphy.com/media/artj92V8o75VPL7AeQ/giphy.gif',
        'Sadness': 'https://media.giphy.com/media/ISOckXUybVfQ4/giphy.gif',
        'Anger': 'https://media.giphy.com/media/12XMGIWtrHBl5e/giphy.gif',
        'Fear': 'https://media.giphy.com/media/32mC2kXYWCsg0/giphy.gif',
        'Surprise': 'https://media.giphy.com/media/5VKbvrjxpVJCM/giphy.gif',
        'Disgust': 'https://media.giphy.com/media/DsdVe5jhHWNC8/giphy.gif',
        'Love': 'https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif',
        'Irritation': 'https://media.giphy.com/media/l4FGGafcOHmrlQxG0/giphy.gif',
        'Dismissive': 'https://media.giphy.com/media/Rhhr8D5mKSX7O/giphy.gif',
        'Passive-Aggressive': 'https://media.giphy.com/media/A3IKIsvG1UjwA/giphy.gif'
    }
    return fallback_gifs.get(emotion, 'https://media.giphy.com/media/3o7TKTDn976rzVgky4/giphy.gif')

def get_mood_description(emotion, confidence):
    """Get description for the detected mood"""
    descriptions = {
        'Joy': "You seem to be in a positive, happy mood! 🌟",
        'Sadness': "It sounds like you might be feeling down or melancholic. 💙",
        'Anger': "There seems to be some frustration or anger in your message. 🔥",
        'Fear': "Your text suggests you might be feeling anxious or worried. 🤗",
        'Surprise': "Your message conveys surprise or amazement! ✨",
        'Disgust': "You seem to be expressing distaste or disapproval. 🤔",
        'Love': "Your text radiates warmth and affection! 💖",
        'Irritation': "You sound frustrated and irritated. That's totally understandable. 😤",
        'Dismissive': "You seem fed up and dismissive. Sometimes people just don't listen! 🙄",
        'Passive-Aggressive': "There's some passive-aggressive energy here. You might be holding back what you really want to say. 😒"
    }
    
    confidence_text = ""
    if confidence > 80:
        confidence_text = " (Very confident)"
    elif confidence > 60:
        confidence_text = " (Fairly confident)"
    else:
        confidence_text = " (Less certain)"
    
    return descriptions.get(emotion, "Neutral mood detected.") + confidence_text
    """Get description for the detected mood"""
    descriptions = {
        'Joy': "You seem to be in a positive, happy mood! 🌟",
        'Sadness': "It sounds like you might be feeling down or melancholic. 💙",
        'Anger': "There seems to be some frustration or anger in your message. 🔥",
        'Fear': "Your text suggests you might be feeling anxious or worried. 🤗",
        'Surprise': "Your message conveys surprise or amazement! ✨",
        'Disgust': "You seem to be expressing distaste or disapproval. 🤔",
        'Love': "Your text radiates warmth and affection! 💖",
        'Irritation': "You sound frustrated and irritated. That's totally understandable. 😤",
        'Dismissive': "You seem fed up and dismissive. Sometimes people just don't listen! 🙄",
        'Passive-Aggressive': "There's some passive-aggressive energy here. You might be holding back what you really want to say. 😒"
    }
    
    confidence_text = ""
    if confidence > 80:
        confidence_text = " (Very confident)"
    elif confidence > 60:
        confidence_text = " (Fairly confident)"
    else:
        confidence_text = " (Less certain)"
    
    return descriptions.get(emotion, "Neutral mood detected.") + confidence_text

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Mood Detector", page_icon="🎭", layout="wide")
    
    st.title("🎭 AI Mood Detector")
    st.markdown("*Analyze emotions and mood from your text using AI!*")
    
    # Load model
    with st.spinner("Loading AI model... (this might take a moment)"):
        emotion_pipeline = load_model()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type text", "Example texts"])
        
        if input_method == "Type text":
            user_text = st.text_area(
                "What's on your mind?", 
                placeholder="Type or paste your text here... (tweets, messages, journal entries, etc.)",
                height=150
            )
        else:
            example_texts = [
                "I'm so excited about my vacation next week! Can't wait to relax on the beach.",
                "This traffic is driving me crazy. I'm going to be late for my important meeting.",
                "I just watched the most beautiful sunset. It made me feel so peaceful and grateful.",
                "I can't believe they cancelled the concert. I was really looking forward to it.",
                "My heart is racing before this job interview. What if I mess up?",
                "That movie was absolutely disgusting. I want my money back.",
                "I love spending time with my family. They mean the world to me."
            ]
            
            selected_example = st.selectbox("Choose an example:", [""] + example_texts)
            user_text = selected_example
            
            if selected_example:
                st.text_area("Selected text:", value=selected_example, height=100, disabled=True)
    
    with col2:
        st.subheader("🎯 Quick Stats")
        if user_text:
            word_count = len(user_text.split())
            char_count = len(user_text)
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
    
    # Analyze button
    if st.button("🔍 Analyze Mood", type="primary", use_container_width=True):
        if user_text:
            with st.spinner("Analyzing your mood..."):
                result = analyze_mood(user_text, emotion_pipeline)
                
                if result:
                    # Display results
                    st.subheader("📊 Mood Analysis Results")
                    
                    # GIF at the top - most prominent
                    with st.spinner("Finding the perfect GIF..."):
                        gif_url = get_gif_for_emotion(result['dominant_emotion'])
                        if not gif_url:
                            gif_url = get_fallback_gif(result['dominant_emotion'])
                        
                        if gif_url:
                            # Center the GIF
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.markdown(f"### 🎬 Your Mood Right Now:")
                                st.image(gif_url, width=300)
                                emoji = get_mood_emoji(result['dominant_emotion'])
                                st.markdown(f"<h3 style='text-align: center;'>{emoji} {result['dominant_emotion']} ({result['confidence']}% confidence)</h3>", unsafe_allow_html=True)
                    
                    # Add some space
                    st.markdown("---")
                    
                    # Main result below GIF
                    description = get_mood_description(result['dominant_emotion'], result['confidence'])
                    st.info(description)
                    
                    # Detailed breakdown
                    st.subheader("📈 Detailed Emotion Breakdown")
                    
                    # Create DataFrame for plotting
                    emotions_df = pd.DataFrame(
                        list(result['all_emotions'].items()),
                        columns=['Emotion', 'Score']
                    ).sort_values('Score', ascending=True)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        emotions_df, 
                        x='Score', 
                        y='Emotion',
                        orientation='h',
                        title="Emotion Confidence Scores",
                        color='Score',
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show all scores
                    st.subheader("🔢 All Emotion Scores")
                    scores_df = pd.DataFrame(
                        list(result['all_emotions'].items()),
                        columns=['Emotion', 'Confidence (%)']
                    ).sort_values('Confidence (%)', ascending=False)
                    
                    st.dataframe(scores_df, use_container_width=True)
                    
                else:
                    st.error("Could not analyze the text. Please try with different text.")
        else:
            st.warning("Please enter some text to analyze!")
    
    with st.expander("🎬 About the GIFs"):
        st.markdown("""
        **GIF Features:**
        - **Real-time**: Fetches GIFs from Giphy API based on your emotion
        - **Variety**: Different GIF for each emotion detection
        - **Family-friendly**: All GIFs are rated G for general audiences
        - **Fallback**: Backup GIFs in case the API is unavailable
        
        *Want your own GIF API key? Get one free at [developers.giphy.com](https://developers.giphy.com/)*
        """)
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This AI mood detector uses a pre-trained transformer model specifically fine-tuned for emotion detection:
        
        - **Model**: DistilRoBERTa fine-tuned on emotion classification
        - **Emotions Detected**: Joy, Sadness, Anger, Fear, Surprise, Disgust, Love
        - **Technology**: Hugging Face Transformers, PyTorch
        - **Confidence**: Shows how certain the AI is about each emotion
        
        The model analyzes the semantic content and emotional context of your text to predict the most likely emotions present.
        """)
    
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - **Length**: Works best with 10-200 words
        - **Context**: More context generally gives better results
        - **Language**: Optimized for English text
        - **Content**: Works well with social media posts, messages, reviews, diary entries
        - **Multiple emotions**: The tool can detect mixed emotions in complex text
        """)

if __name__ == "__main__":
    main()