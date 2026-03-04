import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# Load dataset
df = pd.read_csv("data/anxiety_dataset.csv")

# Convert labels to numbers
label_map = {"low":0,"moderate":1,"high":2}
df["label"] = df["label"].map(label_map)

texts = df["text"].tolist()
labels = df["label"].tolist()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class AnxietyDataset(Dataset):

    def _init_(self,texts,labels):
        self.texts = texts
        self.labels = labels

    def _len_(self):
        return len(self.texts)

    def _getitem_(self,idx):

        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids":encoding["input_ids"].squeeze(),
            "attention_mask":encoding["attention_mask"].squeeze(),
            "labels":torch.tensor(self.labels[idx])
        }

dataset = AnxietyDataset(texts,labels)

loader = DataLoader(dataset,batch_size=2)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)

model.train()

for epoch in range(3):

    for batch in loader:

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

print("Training Complete")

torch.save(model.state_dict(),"bert_anxiety_model.pt")