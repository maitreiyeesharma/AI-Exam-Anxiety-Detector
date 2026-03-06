import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "bert_anxiety_model.pt"

# Use CPU for inference
device = torch.device("cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Move model to CPU
model.to(device)

# Set evaluation mode
model.eval()

print("BERT Anxiety Model Loaded Successfully")