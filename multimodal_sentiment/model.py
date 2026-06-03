import torch
import torch.nn as nn

class MultimodalSentimentModel(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=16, hidden_dim=64):
        super(MultimodalSentimentModel, self).__init__()

        # 1. TEXT BRANCH LAYER
        self.text_embedder = nn.Embedding(vocab_size, embedding_dim)

        # 2. IMAGE BRANCH LAYER
        self.image_projection = nn.Linear(3 * 224 * 224, hidden_dim)

        # 3. CLASSIFIER HEAD
        combined_features_dim = embedding_dim + hidden_dim

        # Output dim is 2 because we have 2 classes (Negative=0, Positive=1)
        self.classifier = nn.Linear(combined_features_dim, 2)

    def forward(self, image, text):
        # image shape: [batch_size, 3, 224, 224]
        # text shape: [batch_size, seq_len]

        # --- Process Image ---
        x_img = image.view(image.size(0), -1)
        x_img = torch.relu(self.image_projection(x_img))

        # --- Process Text ---
        x_txt = self.text_embedder(text)
        x_txt = torch.mean(x_txt, dim=1)

        # --- Fusion Phase ---
        fused_features = torch.cat((x_img, x_txt), dim=1)

        # --- Final Prediction ---
        output = self.classifier(fused_features)
        return output


# --- Verification Block ---
if __name__ == "__main__":
    model = MultimodalSentimentModel()
    print("Model initialized successfully!")

    # Mock a batch of 2 samples to test the forward pass mechanics
    mock_batch_img = torch.randn(2, 3, 224, 224)
    mock_batch_txt = torch.randint(0, 5, (2, 4))

    predictions = model(mock_batch_img, mock_batch_txt)

    print("Output shape from forward pass:", predictions.shape)
    # Expected: torch.Size([2, 2])