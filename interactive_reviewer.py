import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# First, recreate the model class
class ProductReviewClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(ProductReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.softmax(linear_output)

# Create the interactive review function
def analyze_review():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        model = ProductReviewClassifier().to(device)
        model.load_state_dict(torch.load('product_review_classifier.pth', map_location=device))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()
        
        sentiment_map = {0: "Poor", 1: "Average", 2: "Good"}
        
        print("\nReview Sentiment Analyzer")
        print("------------------------")
        print("Enter 'quit' to exit")
        
        while True:
            # Get user input
            review = input("\nEnter your review: ").strip()
            
            # Check for quit command
            if review.lower() == 'quit':
                print("Goodbye!")
                break
            
            # Skip empty input
            if not review:
                print("Please enter a review.")
                continue
            
            # Process the review
            encoding = tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)
                probabilities = outputs[0].tolist()
                
            sentiment = predicted.item()
            
            # Print results
            print("\nResults:")
            print(f"Review: {review}")
            print(f"Sentiment: {sentiment} ({sentiment_map[sentiment]})")
            print(f"Confidence scores:")
            print(f"  Poor: {probabilities[0]*100:.2f}%")
            print(f"  Average:  {probabilities[1]*100:.2f}%")
            print(f"  Good: {probabilities[2]*100:.2f}%")
            
    except FileNotFoundError:
        print("Error: Model file 'product_review_classifier.pth' not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    analyze_review()