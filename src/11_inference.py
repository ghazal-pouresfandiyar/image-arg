"""
Inference Pipeline: Load trained model and generate conclusions
Premise + Visual Features → Conclusion Generation
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('/Users/poures/Desktop/PC/image-arg/models')
DATA_DIR = Path('/Users/poures/Desktop/PC/image-arg/dataset/unified_features')

# Import model classes
import sys
sys.path.insert(0, '/Users/poures/Desktop/PC/image-arg/src')
from argument_model import ArgumentGenerationModel


class ArgumentInference:
    """Generate conclusions from premises and visual features"""
    
    def __init__(self, model_path, vocab_path, device=device):
        """
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary JSON
            device: torch device (cpu/cuda)
        """
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_info = json.load(f)
        
        self.word2idx = vocab_info['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_info['idx2word'].items()}
        self.vocab_size = vocab_info['vocab_size']
        self.device = device
        
        print(f"✓ Vocabulary loaded: {self.vocab_size} tokens")
        
        # Initialize model
        self.model = ArgumentGenerationModel(
            feature_dim=932,
            vocab_size=self.vocab_size,
            hidden_dim=512,
            max_seq_len=150
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded: {model_path}")
    
    def _tokenize(self, text, max_len=150):
        """Convert text to token indices"""
        tokens = [self.word2idx['<START>']]
        words = text.lower().split()[:max_len - 2]
        
        for word in words:
            tokens.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        tokens.append(self.word2idx['<END>'])
        
        while len(tokens) < max_len:
            tokens.append(self.word2idx['<PAD>'])
        
        return np.array(tokens[:max_len])
    
    def _vectorize_facts(self, facts_list, max_len=200):
        """Convert fact list to token indices"""
        facts_tokens = []
        
        for fact_item in facts_list[:5]:  # Top 5 facts
            fact_text = fact_item['fact'] if isinstance(fact_item, dict) else fact_item
            tokens = self._tokenize(fact_text)
            facts_tokens.extend(tokens)
        
        if len(facts_tokens) < max_len:
            facts_tokens.extend([self.word2idx['<PAD>']] * (max_len - len(facts_tokens)))
        
        return np.array(facts_tokens[:max_len])
    
    def generate(self, features, premises_text, facts_list, max_len=150):
        """
        Generate conclusion from premises and features
        
        Args:
            features: (932,) feature vector
            premises_text: String of premises
            facts_list: List of retrieved facts
            max_len: Max generation length
        
        Returns:
            conclusion_text: Generated conclusion string
            confidence: Average confidence score
        """
        with torch.no_grad():
            # Prepare inputs
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, 932)
            premise_tokens = self._tokenize(premises_text)
            premise_tensor = torch.LongTensor(premise_tokens).unsqueeze(0).to(self.device)  # (1, 150)
            fact_tokens = self._vectorize_facts(facts_list)
            fact_tensor = torch.LongTensor(fact_tokens).unsqueeze(0).to(self.device)  # (1, 200)
            
            # Get encoder context
            encoder_context = self.model.encoder(features_tensor, premise_tensor)  # (1, 512)
            
            # Initialize decoder
            h = encoder_context
            c = torch.zeros(1, 512).to(self.device)
            
            # Embed facts
            facts_emb = self.model.decoder.fact_embed(fact_tensor)  # (1, 200, 256)
            
            # Generate tokens
            generated_tokens = [self.word2idx['<START>']]
            confidences = []
            
            current_token = torch.tensor([[self.word2idx['<START>']]], dtype=torch.long).to(self.device)
            
            for t in range(max_len - 1):
                # Get embedding - current_token is (1, 1), squeeze to get scalar
                token_idx = current_token.squeeze(0).squeeze(0)  # Scalar
                embedded = self.model.decoder.embed(token_idx).unsqueeze(0)  # (1, 256)
                
                # LSTM step
                h, c = self.model.decoder.lstm(embedded, (h, c))
                
                # Fact attention
                query = self.model.decoder.attention_query(h)  # (1, 256)
                attention_scores = torch.bmm(facts_emb, query.unsqueeze(2)).squeeze(2)  # (1, 200)
                attention_weights = torch.softmax(attention_scores, dim=1)
                fact_context = torch.bmm(
                    attention_weights.unsqueeze(1),
                    facts_emb
                ).squeeze(1)  # (1, 256)
                
                # Combine
                fact_context_proj = self.model.decoder.fact_to_hidden(fact_context)
                combined = h + fact_context_proj
                
                # Output logits
                logits = self.model.decoder.output_layer(combined)  # (1, vocab_size)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1)[0].item()
                confidences.append(confidence)
                
                next_token = torch.argmax(logits, dim=-1)
                token_id = next_token.item()
                
                generated_tokens.append(token_id)
                
                # Stop on <END>
                if token_id == self.word2idx['<END>']:
                    break
                
                current_token = next_token.unsqueeze(0)
            
            # Convert tokens to text
            conclusion_text = self._detokenize(generated_tokens)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return conclusion_text, avg_confidence
    
    def _detokenize(self, token_ids):
        """Convert token indices to text"""
        words = []
        for token_id in token_ids:
            if token_id in [0, self.word2idx['<PAD>'], self.word2idx['<START>']]:
                continue
            if token_id == self.word2idx['<END>']:
                break
            word = self.idx2word.get(token_id, '<UNK>')
            words.append(word)
        
        return ' '.join(words)


# ============================================================================
# Inference Demo
# ============================================================================

def demo_inference():
    """Run inference on sample images"""
    
    print("\n" + "="*70)
    print("INFERENCE PIPELINE: ARGUMENT GENERATION")
    print("="*70)
    
    # Initialize inference
    print("\n[1/3] Loading inference engine...")
    inferencer = ArgumentInference(
        model_path=MODEL_DIR / 'argument_model.pt',
        vocab_path=MODEL_DIR / 'vocabulary.json',
        device=device
    )
    
    # Load data
    print("\n[2/3] Loading dataset...")
    data = np.load(DATA_DIR / 'unified_features.npz')
    features = data['unified_features']  # (58, 932)
    
    with open(DATA_DIR / 'targets.json', 'r') as f:
        targets = json.load(f)
    premises_list = targets['premises']
    conclusions_list = targets['conclusions']
    
    with open(DATA_DIR / 'retrieved_facts.json', 'r') as f:
        facts_data = json.load(f)
    facts_list = [item['retrieved_facts'] for item in facts_data]
    
    print(f"✓ Loaded {len(features)} samples")
    
    # Generate on samples
    print("\n[3/3] Generating conclusions...\n")
    
    results = []
    sample_indices = [0, 10, 20, 30, 40, 50]  # Sample 6 images
    
    for idx in sample_indices:
        print(f"━" * 70)
        print(f"Sample {idx}:")
        print(f"━" * 70)
        
        # Get data
        image_features = features[idx]
        premises_text = ' '.join(premises_list[idx]) if premises_list[idx] else "No premise"
        ground_truth = conclusions_list[idx][0] if conclusions_list[idx] else "No conclusion"
        facts = facts_list[idx]
        
        # Generate
        generated, confidence = inferencer.generate(
            image_features,
            premises_text,
            facts,
            max_len=150
        )
        
        print(f"Premise: {premises_text[:80]}...")
        print(f"\nGround Truth: {ground_truth}")
        print(f"\nGenerated:    {generated}")
        print(f"Confidence:   {confidence:.4f}")
        
        results.append({
            'index': idx,
            'premise': premises_text,
            'ground_truth': ground_truth,
            'generated': generated,
            'confidence': confidence
        })
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving inference results...")
    
    output_dir = Path('/Users/poures/Desktop/PC/image-arg/inference_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'generated_conclusions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir / 'generated_conclusions.json'}")
    
    return results, inferencer


if __name__ == '__main__':
    results, inferencer = demo_inference()
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"\nGenerated {len(results)} sample conclusions")
