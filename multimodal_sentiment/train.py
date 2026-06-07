import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ProductDataset
from model import MultimodalSentimentModel

def train_model():
   
    EPOCHS = 5
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    
  
    dataset = ProductDataset(num_samples=20)
  
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultimodalSentimentModel()
    
    
    criterion = nn.CrossEntropyLoss()
    
  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- Starting Multimodal Training Loop ---")
    
   
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
       
        for batch_idx, (images, texts, labels) in enumerate(dataloader):
            
            optimizer.zero_grad()
            
       
            predictions = model(images, texts)
            
           
            loss = criterion(predictions, labels)
            
 
            loss.backward()
            
 
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")
        
    print("--- Training Completed Successfully! ---")

if __name__ == "__main__":
    train_model()