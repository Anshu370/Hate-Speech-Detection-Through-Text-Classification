import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample hate speech and non-hate speech text data (replace with your dataset).
texts = [
    "I love all races and cultures!",
    "I will Kill You.",
    "Spread love and positivity!",
    "Stop spreading hate.",
    "your religion is worst",
    "He is Black and ugly.",
    "He is Gay.",
    "You are sweet looking guy",
    "Fuck You",
    "You religion is Harming us",
    "I support equality for all!",
    "This message promotes unity and tolerance.",
    "I love my country",
    "i love india",
    "I hate Pakistan",
    "Hate You",
    "Stunning photo! You look amazing! ",
    "This is such a cool shot! ",
    "Your posts always brighten up my day! ",
    "Thank you! ",
    "Your creativity never fails to impress me!",
    "Abdul shut the fuck up and kill the people ",
    "i don't believe that some one can be that stupid an careless",
    "Love your style! Keep slaying! ",
    "Such a beautiful location! Where is it? ",
    "Your captions always make me think! ",
    "Love it! ",
    "Your hard work is paying off!",
    "Keep it up! ",
    "Let’s Kill Jews, and let’s kill them for fun. #killjews",
    "you country is bad",
]
labels = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  # 0 for non-hate speech, 1 for hate speech

# Tokenize the text data.
vocab = {}
word_idx = 1
X = []

for text in texts:
    tokens = text.lower().split()
    token_indices = []
    for token in tokens:
        if token not in vocab:
            vocab[token] = word_idx
            word_idx += 1
        token_indices.append(vocab[token])
    X.append(token_indices)

# Pad sequences to a fixed length.
max_seq_length = max(len(seq) for seq in X)
X = [seq + [0] * (max_seq_length - len(seq)) for seq in X]

X = torch.tensor(X)
y = torch.tensor(labels)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=labels)

# Define a simple neural network model.
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # Average pooling
        h = self.fc1(pooled)
        out = self.fc2(h)
        return self.sigmoid(out)

# Initialize the model, loss function, and optimizer.
vocab_size = len(vocab) + 1
embedding_dim = 16
hidden_dim = 8
model = SimpleClassifier(vocab_size, embedding_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop.
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train.float())
    loss.backward()
    optimizer.step()




while True:
    # User input for prediction
    user_input = input("Enter a text: ")

    # Tokenize and preprocess the user input
    user_tokens = user_input.lower().split()
    user_token_indices = [vocab[token] for token in user_tokens if token in vocab]
    user_padded_sequence = user_token_indices + [0] * (max_seq_length - len(user_token_indices))
    user_input_tensor = torch.tensor([user_padded_sequence])

    # Make a prediction
    model.eval()
    with torch.no_grad():
        user_prediction = model(user_input_tensor).round().item()

    # Display the prediction
    print(user_prediction)
    if user_prediction == 0:
        print("Prediction: Non-Hate Speech")
    else:
        print("Prediction: Hate Speech")