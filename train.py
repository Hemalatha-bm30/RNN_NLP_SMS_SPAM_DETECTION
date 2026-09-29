import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import build_vocab, encode, pad_sequence
from model import SMSRNN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv("data/SVL_SMS_Spam_Practice_Dataset.csv")
label_map = {"ham": 0, "spam": 1}
y = df["label"].map(label_map).values

train_texts, test_texts, y_train, y_test = train_test_split(
    df["message"].tolist(), y, test_size=0.25, random_state=SEED, stratify=y
)

vocab = build_vocab(train_texts)
MAX_LEN = 20

def make_tensor(texts):
    encoded = [pad_sequence(encode(t, vocab), MAX_LEN) for t in texts]
    return torch.tensor(encoded, dtype=torch.long)

X_train = make_tensor(train_texts)
X_test = make_tensor(test_texts)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SMSRNN(len(vocab)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

print("\nAccuracy:", accuracy_score(y_test.numpy(), preds))
print(classification_report(y_test.numpy(), preds, target_names=["ham", "spam"], zero_division=0))

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "max_len": MAX_LEN,
    "label_map": label_map
}, "models/sms_rnn.pt")

print("Saved: models/sms_rnn.pt")