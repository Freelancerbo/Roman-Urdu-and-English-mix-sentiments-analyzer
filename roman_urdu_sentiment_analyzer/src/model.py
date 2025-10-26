# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RomanUrduEnglishPreprocessor:
    def __init__(self):
        # Roman Urdu stop words
        self.roman_urdu_stopwords = {
            'hai', 'hain', 'ho', 'main', 'mera', 'meri', 'mese', 'ko', 'ka', 'ki', 
            'ke', 'ne', 'par', 'phir', 'se', 'tak', 'taraf', 'wo', 'waha', 'ye', 'yaha',
            'aur', 'awr', 'bohat', 'bohot', 'zaroori', 'lekin', 'kuch', 'kyun', 'to', 'bhi',
            'tha', 'thi', 'the', 'apna', 'apne', 'apni', 'hum', 'tum', 'un', 'unka', 'unki',
            'unke', 'iska', 'iski', 'iske', 'uska', 'uski', 'uske', 'aik', 'do', 'teen',
            'mein', 'tumhara', 'kya', 'nahi', 'thi', 'rakh'
        }
        
        # English stop words
        self.english_stopwords = set(stopwords.words('english'))
        
        # Combined stop words
        self.all_stopwords = self.roman_urdu_stopwords.union(self.english_stopwords)
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if text is None:
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits but keep Roman Urdu specific characters
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.all_stopwords]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """Main preprocessing function"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class MixedSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.preprocessor = RomanUrduEnglishPreprocessor()
        self.is_trained = False
    
    def train(self, texts, labels):
        """Train the sentiment analyzer"""
        # Preprocess texts
        cleaned_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
        
        print("Model training completed!")
        return self
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if not self.is_trained:
            raise Exception("Model not trained yet! Call train() first.")
        
        # Preprocess text
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return prediction, probability
    
    def predict_sentiment(self, text):
        """Get detailed sentiment analysis"""
        prediction, probability = self.predict(text)
        confidence = max(probability)
        
        # Get class names
        classes = self.model.classes_
        probabilities_dict = {cls: prob for cls, prob in zip(classes, probability)}
        
        result = {
            'text': text,
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': probabilities_dict
        }
        return result
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self