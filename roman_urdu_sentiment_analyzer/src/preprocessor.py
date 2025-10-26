# src/preprocessor.py
import re
import pandas as pd
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
        if pd.isna(text):
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