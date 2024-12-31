import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ENCODER: CNN (Pretrained ResNet50)
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
        self.fc = nn.Linear(resnet.fc.in_features, 512)  # Reduce feature dimensions to 512
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.resnet(x)
        features = torch.flatten(features, 1)
        features = self.relu(self.fc(features))
        return features

# 2. DECODER: RNN (LSTM-based)
# DECODER: RNN (LSTM-based)
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

        # Add a linear layer to project encoder features to embed_size
        self.feature_projection = nn.Linear(512, embed_size)

    def forward(self, features, captions):
        # Project features to embed_size
        projected_features = self.feature_projection(features)

        # Embeddings for captions
        embeddings = self.embed(captions[:, :-1])  # Exclude <end> token

        # Concatenate projected features and embeddings
        inputs = torch.cat((projected_features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs


# 3. CUSTOM DATASET
class CustomDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_caption_length=20):
        self.image_dir = image_dir
        self.transform = transform
        self.max_caption_length = max_caption_length

        # Read captions
        with open(captions_file, 'r') as f:
            lines = f.readlines()

        self.captions_dict = {}
        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                image_name, caption = parts
                if image_name not in self.captions_dict:
                    self.captions_dict[image_name] = []
                self.captions_dict[image_name].append(caption)

        # Build vocabulary
        self.vocab = self.build_vocab()

        # Image names
        self.image_names = list(self.captions_dict.keys())

    def build_vocab(self):
        all_captions = [caption for captions in self.captions_dict.values() for caption in captions]
        words = [word.lower() for caption in all_captions for word in caption.split()]
        word_counts = Counter(words)

        vocab = {word: idx + 4 for idx, (word, _) in enumerate(word_counts.items())}
        vocab['<pad>'] = 0
        vocab['<start>'] = 1
        vocab['<end>'] = 2
        vocab['<unk>'] = 3
        return vocab

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        captions = self.captions_dict[image_name]
        caption = random.choice(captions)  # Select a random caption

        if self.transform:
            image = self.transform(image)

        caption_indices = self.convert_caption_to_indices(caption)
        return image, torch.tensor(caption_indices, dtype=torch.long)

    def convert_caption_to_indices(self, caption):
        tokens = ['<start>'] + caption.lower().split() + ['<end>']
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return self.pad_or_truncate(indices)

    def pad_or_truncate(self, indices):
        if len(indices) < self.max_caption_length:
            indices += [self.vocab['<pad>']] * (self.max_caption_length - len(indices))
        return indices[:self.max_caption_length]

# 4. DATA PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
image_dir = 'C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/Images'
captions_file = 'C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/captions.txt'

train_dataset = CustomDataset(image_dir, captions_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. MODEL INITIALIZATION
vocab_size = len(train_dataset.vocab)
encoder = EncoderCNN().to(DEVICE)
decoder = DecoderRNN(vocab_size, embed_size=256, hidden_size=512).to(DEVICE)

# 6. LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<pad>'])
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# 7. TRAINING LOOP
def train_model(encoder, decoder, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for images, captions in train_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions)

            # Align the outputs and target sequence lengths
            target = captions[:, 1:]  # Shift right for target
            outputs = outputs[:, :target.size(1), :]  # Match target length

            loss = criterion(outputs.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print("Training complete!")

# Train the model
train_model(encoder, decoder, train_loader, criterion, optimizer, num_epochs=10)
