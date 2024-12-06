import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the dataset
with open("wikipedia_scraped_data.json", "r", encoding='utf-8') as file:
    data = json.load(file)

# Prepare the dataset
records = []
for topic, articles in data.items():
    for article in articles:
        title = article["title"] if isinstance(article["title"], str) else ""
        summary = article["summary"] if isinstance(article["summary"], str) else ""
        records.append({"text": title + " " + summary, "topic": topic})

df = pd.DataFrame(records)

# Encode the topic labels
topic_labels = {topic: idx for idx, topic in enumerate(df["topic"].unique())}
df["label"] = df["topic"].map(topic_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Define a custom dataset class
class TopicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create datasets
train_dataset = TopicDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len=128)
test_dataset = TopicDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_len=128)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(topic_labels)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./topic_model")
tokenizer.save_pretrained("./topic_model")
