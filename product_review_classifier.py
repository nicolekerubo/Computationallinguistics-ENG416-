import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

class ProductReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ProductReviewClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.1):
        super(ProductReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = output.pooler_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.softmax(linear_output)

def predict_sentiment(text, model, tokenizer, device):
    """
    Predict sentiment for a given text using the trained model.
    
    Args:
        text (str): The review text to classify
        model (ProductReviewClassifier): The trained model
        tokenizer (BertTokenizer): The BERT tokenizer
        device (torch.device): The device to run inference on
    
    Returns:
        tuple: (predicted_class, confidence_score)
    """
    model.eval()
    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    text = ' '.join(text.split())
    
    encoding = tokenizer.encode_plus(
        text,
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
        probabilities = outputs
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {total_loss/len(train_loader)}')
        print(f'Average Validation Loss: {val_loss/len(val_loader)}')
        print(f'Validation Accuracy: {100 * correct/total}%')
        print('-' * 50)
        
        model.train()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and preprocess your data
    # This is a placeholder - replace with your actual data loading
    df = pd.DataFrame({
        'review': ['Great product!', 'Terrible experience', 'Average item'],
        'label': [2, 0, 1]  # Example labels: 0 (negative), 1 (neutral), 2 (positive)
    })
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].values, 
        df['label'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets
    train_dataset = ProductReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProductReviewDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = ProductReviewClassifier(n_classes=3).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Save the model
    torch.save(model.state_dict(), 'product_review_classifier.pth')

if __name__ == "__main__":
    main()