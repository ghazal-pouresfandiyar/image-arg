"""
Argument Generation Model: Premise + Visual Features → Conclusion
Encoder-Decoder with Fact-Augmented Attention
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_DIR = Path('/Users/poures/Desktop/PC/image-arg/dataset/unified_features')
MODEL_DIR = Path('/Users/poures/Desktop/PC/image-arg/models')
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# Dataset Class
# ============================================================================

class ClimateArgumentDataset(Dataset):
    """Load unified features, premises, conclusions, and facts"""
    
    def __init__(self, features_path, targets_path, retrieved_facts_path, vocab_size=10000, max_seq_len=150):
        # Load unified features
        data = np.load(features_path)
        self.features = torch.FloatTensor(data['unified_features']).to(device)  # (58, 932)
        
        # Load targets (premises, conclusions)
        with open(targets_path, 'r') as f:
            targets = json.load(f)
        
        # Load retrieved facts
        with open(retrieved_facts_path, 'r') as f:
            facts_list = json.load(f)
        
        # Convert facts list to dict indexed by image_id
        self.facts = {}
        for item in facts_list:
            self.facts[item['id']] = item['retrieved_facts']
        
        self.premises = targets['premises']  # List of lists
        self.conclusions = targets['conclusions']  # List of lists
        
        # Build vocabulary from conclusions and premises
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = self._build_vocabulary()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        print(f"✓ Dataset loaded: {len(self.features)} samples")
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
    
    def _build_vocabulary(self):
        """Build word2idx mapping from conclusions and premises"""
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        word_freq = Counter()
        
        # Count word frequencies
        for conclusion_list in self.conclusions:
            for conclusion in conclusion_list:
                words = conclusion.lower().split()
                word_freq.update(words)
        
        for premise_list in self.premises:
            for premise in premise_list:
                words = premise.lower().split()
                word_freq.update(words)
        
        # Add top vocab_size - 4 words to vocab
        idx = 4
        for word, freq in word_freq.most_common(self.vocab_size - 4):
            vocab[word] = idx
            idx += 1
        
        return vocab
    
    def _tokenize(self, text):
        """Convert text to token indices"""
        tokens = [self.word2idx['<START>']]
        words = text.lower().split()[:self.max_seq_len - 2]
        
        for word in words:
            tokens.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        tokens.append(self.word2idx['<END>'])
        
        # Pad to max_seq_len
        while len(tokens) < self.max_seq_len:
            tokens.append(self.word2idx['<PAD>'])
        
        return np.array(tokens[:self.max_seq_len])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get image features
        features = self.features[idx]  # (932,)
        
        # Get premises (concatenate all premises for this image)
        premise_text = ' '.join(self.premises[idx]) if self.premises[idx] else "no premise"
        premise_tokens = torch.LongTensor(self._tokenize(premise_text))  # (150,)
        
        # Get conclusion (take first one if multiple)
        conclusion_text = self.conclusions[idx][0] if self.conclusions[idx] else "no conclusion"
        conclusion_tokens = torch.LongTensor(self._tokenize(conclusion_text))  # (150,)
        
        # Get retrieved facts
        image_id = str(idx)
        facts_list = self.facts.get(image_id, [])
        
        # Convert facts to tokens and concatenate
        facts_tokens = []
        for fact_item in facts_list[:5]:  # Top 5 facts
            fact_text = fact_item['fact'] if isinstance(fact_item, dict) else fact_item
            tokens = self._tokenize(fact_text)
            facts_tokens.extend(tokens)
        
        # Pad or truncate facts to 200 tokens
        if len(facts_tokens) < 200:
            facts_tokens.extend([self.word2idx['<PAD>']] * (200 - len(facts_tokens)))
        facts_tokens = facts_tokens[:200]
        facts_embedding = torch.LongTensor(facts_tokens)  # (200,)
        
        return {
            'features': features,        # (932,)
            'premises': premise_tokens,  # (150,)
            'conclusions': conclusion_tokens,  # (150,)
            'facts': facts_embedding     # (200,)
        }


# ============================================================================
# Model Components
# ============================================================================

class ArgumentEncoder(nn.Module):
    """Encode visual features + premises → 512-dim representation"""
    
    def __init__(self, feature_dim=932, hidden_dim=512, premise_vocab_size=10000):
        super().__init__()
        
        # Visual feature encoder: 932 → 768 → 512
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU()
        )
        
        # Premise encoder: embed + BiLSTM
        self.premise_embed = nn.Embedding(premise_vocab_size, 128)
        self.premise_lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
            dropout=0.0  # Set to 0 for single layer
        )
        
        # Fusion: concatenate visual(512) + premise(512) → 512
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, features, premises):
        """
        Args:
            features: (batch, 932)
            premises: (batch, 150)
        Returns:
            context: (batch, 512)
        """
        # Encode visual features
        visual_repr = self.feature_encoder(features)  # (batch, 512)
        
        # Encode premises
        premise_emb = self.premise_embed(premises)  # (batch, 150, 128)
        premise_lstm_out, (h, c) = self.premise_lstm(premise_emb)  # BiLSTM
        # Take final hidden state from both directions
        premise_repr = torch.cat([h[0], h[1]], dim=-1)  # (batch, 512)
        
        # Fuse visual and premise representations
        fused = torch.cat([visual_repr, premise_repr], dim=-1)  # (batch, 1024)
        context = self.fusion(fused)  # (batch, 512)
        
        return context


class FactAugmentedDecoder(nn.Module):
    """Decode with LSTM + Fact Attention to generate conclusions"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, 
                 fact_dim=200, max_seq_len=150):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM decoder
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        
        # Fact attention
        self.fact_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention_query = nn.Linear(hidden_dim, embedding_dim)
        self.fact_to_hidden = nn.Linear(embedding_dim, hidden_dim)  # Project fact context to hidden dim
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, encoder_context, facts, target_tokens=None, teacher_force_ratio=0.5):
        """
        Args:
            encoder_context: (batch, 512)
            facts: (batch, 200)
            target_tokens: (batch, 150) if training
            teacher_force_ratio: probability of using teacher forcing
        Returns:
            logits: (batch, max_seq_len, vocab_size)
        """
        batch_size = encoder_context.size(0)
        
        # Initialize decoder state from encoder context
        h = encoder_context  # (batch, 512)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)  # (batch, 512)
        
        # Embed facts for attention
        facts_emb = self.fact_embed(facts)  # (batch, 200, embedding_dim)
        
        # Decoder loop
        outputs = []
        
        # Start token
        current_token = torch.full((batch_size,), 1, dtype=torch.long).to(device)  # <START> = 1
        
        for t in range(self.max_seq_len):
            # Get embedding
            embedded = self.embed(current_token)  # (batch, embedding_dim)
            
            # LSTM step
            h, c = self.lstm(embedded, (h, c))  # h, c: (batch, 512)
            
            # Fact attention
            query = self.attention_query(h)  # (batch, embedding_dim)
            attention_scores = torch.bmm(facts_emb, query.unsqueeze(2)).squeeze(2)  # (batch, 200)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, 200)
            fact_context = torch.bmm(
                attention_weights.unsqueeze(1), 
                facts_emb
            ).squeeze(1)  # (batch, embedding_dim)
            
            # Project fact context to hidden dimension and combine with decoder output
            fact_context_proj = self.fact_to_hidden(fact_context)  # (batch, hidden_dim)
            combined = h + fact_context_proj  # (batch, hidden_dim)
            
            # Output logits
            logits = self.output_layer(combined)  # (batch, vocab_size)
            outputs.append(logits)
            
            # Decide next token
            if target_tokens is not None and t < target_tokens.size(1):
                # Teacher forcing
                if torch.rand(1).item() < teacher_force_ratio:
                    current_token = target_tokens[:, t]
                else:
                    current_token = torch.argmax(logits, dim=1)
            else:
                current_token = torch.argmax(logits, dim=1)
        
        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # (batch, max_seq_len, vocab_size)
        return logits


class ArgumentGenerationModel(nn.Module):
    """Full encoder-decoder model for argument generation"""
    
    def __init__(self, feature_dim=932, vocab_size=10000, hidden_dim=512, max_seq_len=150):
        super().__init__()
        
        self.encoder = ArgumentEncoder(feature_dim, hidden_dim, vocab_size)
        self.decoder = FactAugmentedDecoder(vocab_size, 256, hidden_dim, 200, max_seq_len)
    
    def forward(self, features, premises, facts, conclusions=None, teacher_force_ratio=0.5):
        """
        Args:
            features: (batch, 932)
            premises: (batch, 150)
            facts: (batch, 200)
            conclusions: (batch, 150) for training
            teacher_force_ratio: for teacher forcing during training
        Returns:
            logits: (batch, 150, vocab_size)
        """
        encoder_context = self.encoder(features, premises)
        logits = self.decoder(encoder_context, facts, conclusions, teacher_force_ratio)
        return logits


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        features = batch['features'].to(device)
        premises = batch['premises'].to(device)
        conclusions = batch['conclusions'].to(device)
        facts = batch['facts'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features, premises, facts, conclusions, teacher_force_ratio=0.7)
        
        # Compute loss (ignore padding tokens)
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
            conclusions.view(-1)  # (batch*seq_len,)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(num_epochs=5, batch_size=4, learning_rate=1e-3):
    """Train the argument generation model"""
    
    print("\n" + "="*70)
    print("TRAINING ARGUMENT GENERATION MODEL")
    print("="*70)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = ClimateArgumentDataset(
        features_path=DATA_DIR / 'unified_features.npz',
        targets_path=DATA_DIR / 'targets.json',
        retrieved_facts_path=DATA_DIR / 'retrieved_facts.json',
        vocab_size=10000,
        max_seq_len=150
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    print("\n[2/3] Initializing model...")
    model = ArgumentGenerationModel(
        feature_dim=932,
        vocab_size=len(dataset.word2idx),
        hidden_dim=512,
        max_seq_len=150
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {total_params:,} trainable parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n[3/3] Training model...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    # Save model and vocabulary
    print("\n✓ Training complete! Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': dataset.word2idx,
        'idx2word': dataset.idx2word
    }, MODEL_DIR / 'argument_model.pt')
    
    # Save vocabulary
    vocab_info = {
        'word2idx': dataset.word2idx,
        'idx2word': {str(k): v for k, v in dataset.idx2word.items()},
        'vocab_size': len(dataset.word2idx),
        'total_params': total_params,
        'feature_dim': 932,
        'hidden_dim': 512,
        'max_seq_len': 150
    }
    
    with open(MODEL_DIR / 'vocabulary.json', 'w') as f:
        json.dump(vocab_info, f, indent=2)
    
    print(f"✓ Model saved to: {MODEL_DIR / 'argument_model.pt'}")
    print(f"✓ Vocabulary saved to: {MODEL_DIR / 'vocabulary.json'}")
    
    return model, dataset


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Train model
    model, dataset = train_model(num_epochs=5, batch_size=4, learning_rate=1e-3)
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel architecture:")
    print(f"  Encoder: Visual (932→512) + Premise BiLSTM (→512) + Fusion")
    print(f"  Decoder: LSTM with Fact Attention (→vocab)")
    print(f"\nNext steps:")
    print(f"  1. Implement inference: Load model and generate conclusions")
    print(f"  2. Evaluate: Compare with ground truth using BLEU/ROUGE")
    print(f"  3. Deploy: Create API for web interface")
