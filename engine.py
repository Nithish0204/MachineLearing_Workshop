import torch # pip install torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification # pip install transformers

class SentimentEngine:
    """
    A Production-ready wrapper for a Sentiment Analysis Model.
    This encapsulates the model loading and inference logic.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        # Trigger immediate loading on instantiation
        self.load_model()

    def load_model(self):
        """Loads the model and tokenizer into memory/GPU."""
        print(f"--- Loading model: {self.model_name} on {self.device} ---")
        
        # In production, we use the 'pipeline' abstraction for optimized inference
        self.classifier = pipeline(
            "sentiment-analysis", 
            model=self.model_name, 
            device=0 if self.device == "cuda" else -1
        )
        print("--- Model Loaded Successfully ---")

    def predict(self, text: str):
        """
        Performs inference on a single string.
        Returns: Dict containing label and confidence score.
        """
        if not self.classifier:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Simple cleanup/validation
        if not text.strip():
            return {"error": "Empty input text"}

        result = self.classifier(text)[0]
        return {
            "sentiment": result['label'],
            "confidence": round(result['score'], 4),
            "engine": self.model_name
        }

    # def predict_batch(self, texts: list[str]):
    #     """
    #     Performs batch inference (Crucial for Production performance).
    #     """
    #     return self.classifier(texts)

if __name__ == "__main__":

    engine = SentimentEngine()
    
    sample_text = "he was crying because he got placement in a mnc"
    prediction = engine.predict(sample_text)
    
    print(f"\nTest Result:\nText: {sample_text}\nResponse: {prediction}")