import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from product_review_classifier import ProductReviewClassifier, ProductReviewDataset, predict_sentiment
import numpy as np
from torch.nn.utils import clip_grad_norm_

# Significantly expanded dataset with more diverse examples
reviews = [
    # Poor reviews (0)
    "Terrible quality, completely fell apart after one use.",
    "Worst purchase ever, absolute waste of money!",
    "Very disappointed, would not recommend to anyone.",
    "Poor quality control, product arrived damaged.",
    "Complete waste of time and money, avoid at all costs.",
    "Horrible customer service and defective product.",
    "The worst product I've ever bought, don't waste your money.",
    "Broke within a week, very poor quality.",
    "Extremely dissatisfied with this purchase.",
    "Really disappointed, doesn't work as advertised.",
    "Product is a complete scam, stay away!",
    "Absolutely horrible, wouldn't give it even one star.",
    "Save your money, this is pure garbage.",
    "Defective product and terrible customer support.",
    "Total disappointment, nothing works as described.",
    "Cheap materials, poorly made, avoid this product.",
    "Returns process was a nightmare, terrible experience.",
    "Completely useless product, waste of time.",
    "Worst customer service I've ever experienced.",
    "Don't buy this, it's a complete ripoff.",
    
    # Average reviews (1)
    "It's okay, nothing special but gets the job done.",
    "Average product, meets basic requirements.",
    "Decent quality for the price, but nothing extraordinary.",
    "Not bad, but also not great.",
    "Could be better, but works fine for basic needs.",
    "Medium quality, acceptable for occasional use.",
    "It's alright, serves its purpose.",
    "Neither impressed nor disappointed.",
    "Basic functionality, average performance.",
    "Does what it's supposed to, nothing more.",
    "Reasonable quality for the price point.",
    "Middle of the road product, works adequately.",
    "Fair value, but don't expect anything special.",
    "Acceptable performance, nothing outstanding.",
    "Basic product that meets minimal expectations.",
    "Average build quality, serves its purpose.",
    "Not amazing, but not terrible either.",
    "Decent enough for occasional use.",
    "Standard quality, nothing to write home about.",
    "Works as expected, nothing more or less.",
    
    # Good reviews (2) #
    "Excellent product, exceeded all my expectations!",
    "Love it! Best purchase I've made this year.",
    "Outstanding quality and great value for money.",
    "Highly recommend, works perfectly!",
    "Amazing product, couldn't be happier!",
    "Great quality and excellent customer service.",
    "Perfect solution for my needs, very satisfied.",
    "Fantastic product, worth every penny.",
    "Super happy with this purchase, works great!",
    "Really impressed, definitely recommend!",
    "Absolutely fantastic quality and performance!",
    "Outstanding value, exceeds expectations!",
    "Best product in its category, highly satisfied!",
    "Incredible quality, perfect in every way!",
    "Exceptional product, worth every penny!",
    "Superb quality and amazing customer service!",
    "Brilliant product, exactly what I needed!",
    "Five stars, would definitely buy again!",
    "Phenomenal product, exceeded expectations!",
    "Top-notch quality, absolutely love it!"
]

# Balanced labels #
labels = [0] * 20 + [1] * 20 + [2] * 20

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def custom_train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_accuracy = 0
    early_stopping = EarlyStopping(patience=3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    
    for epoch in range(num_epochs):
        # Training phase #
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping #
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase #
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'product_review_classifier.pth', _use_new_zipfile_serialization=True)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': reviews,
        'label': labels
    })
    
    # Calculate class weights for balanced learning
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split data with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].values, 
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = ProductReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProductReviewDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductReviewClassifier(n_classes=3).to(device)
    
    # Define weighted loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Train model
    custom_train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Load best model
    model.load_state_dict(torch.load('product_review_classifier.pth', weights_only=True))
    
    # Test the model
    print("\nTesting model with new reviews:")
    sentiment_map = {0: "poor", 1: "average", 2: "good"}
    
    test_reviews = [
        "This product is absolutely terrible, broke immediately.",
        "It's a decent product, works as expected.",
        "Amazing quality, best purchase ever!",
        "Not great, not terrible, just okay.",
        "Complete waste of money, very disappointed.",
        # Additional test cases
        "This is the worst purchase I've ever made!",
        "Pretty good overall, with some minor issues.",
        "Exceptional quality and service!",
        "Mediocre at best, wouldn't buy again.",
        "Total garbage, avoid at all costs!"
    ]
    
    model.eval()
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, model, tokenizer, device)
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {sentiment_map[sentiment]} (confidence: {confidence:.2f})")