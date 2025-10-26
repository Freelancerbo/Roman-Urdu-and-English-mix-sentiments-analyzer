# src/predictor.py
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            self.model = joblib.load('models/sentiment_model.pkl')
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_sentiment(self, text):
        """Predict sentiment for given text"""
        if self.model is None or self.vectorizer is None:
            print("⚠️ Model not loaded. Please train first.")
            return None
            
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return {
            'text': text,
            'sentiment': prediction,
            'confidence': max(probability),
            'probabilities': dict(zip(self.model.classes_, probability))
        }
    
    def interactive_mode(self):
        """Start interactive prediction mode"""
        print("\n" + "="*50)
        print("🎯 ROMAN URDU SENTIMENT ANALYZER - INTERACTIVE MODE")
        print("="*50)
        print("Type your text in Roman Urdu/English mix")
        print("Type 'quit' or 'exit' to stop")
        print("="*50)
        
        while True:
            user_input = input("\n💬 Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                print("⚠️ Please enter some text")
                continue
                
            # Predict sentiment
            result = self.predict_sentiment(user_input)
            
            if result:
                print(f"\n📊 Result:")
                print(f"   Text: {result['text']}")
                print(f"   Sentiment: {result['sentiment'].upper()}")
                print(f"   Confidence: {result['confidence']:.2%}")
                print(f"   Probabilities:")
                for sentiment, prob in result['probabilities'].items():
                    print(f"     - {sentiment}: {prob:.2%}")

def main():
    predictor = SentimentPredictor()
    
    if predictor.load_model():
        predictor.interactive_mode()
    else:
        print("❌ Please train the model first: python main.py")

if __name__ == "__main__":
    main()