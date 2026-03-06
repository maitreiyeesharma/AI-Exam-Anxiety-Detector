import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Disable GPU (use CPU)
device = torch.device("cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Load trained weights (if available)
try:
    model.load_state_dict(torch.load("model/bert_anxiety_model.pt", map_location=device))
except:
    print("Warning: Trained model file not found, using base BERT model")

# Set model to evaluation mode
model.eval()

# Move model to CPU
model.to(device)