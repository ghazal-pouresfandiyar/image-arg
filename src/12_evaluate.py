"""
Evaluation Pipeline: Compute BLEU/ROUGE metrics for generated arguments
Compare generated conclusions vs ground truth
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import re

# ============================================================================
# Metric Computation
# ============================================================================

class BleuScore:
    """Compute BLEU score (1-4 gram)"""
    
    @staticmethod
    def compute(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Compute BLEU score
        Args:
            reference: Reference string (ground truth)
            hypothesis: Hypothesis string (generated)
            weights: Tuple of weights for 1-4 grams
        """
        def _get_ngrams(text, n):
            """Extract n-grams from text"""
            tokens = text.lower().split()
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        
        # Get n-grams
        ref_ngrams = [_get_ngrams(reference, n) for n in range(1, 5)]
        hyp_ngrams = [_get_ngrams(hypothesis, n) for n in range(1, 5)]
        
        # Compute precisions
        precisions = []
        for n in range(4):
            if sum(hyp_ngrams[n].values()) == 0:
                precisions.append(0.0)
            else:
                matches = sum((hyp_ngrams[n] & ref_ngrams[n]).values())
                total = sum(hyp_ngrams[n].values())
                precisions.append(matches / total if total > 0 else 0.0)
        
        # Brevity penalty
        hyp_len = len(hypothesis.lower().split())
        ref_len = len(reference.lower().split())
        
        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        # Weighted geometric mean
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_precisions = [np.log(p) for p in precisions]
            weighted_sum = sum(w * lp for w, lp in zip(weights, log_precisions))
            bleu = bp * np.exp(weighted_sum)
        
        return bleu


class RougeScore:
    """Compute ROUGE score (ROUGE-L)"""
    
    @staticmethod
    def _lcs_length(s1, s2):
        """Compute longest common subsequence length"""
        m, n = len(s1), len(s2)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
        
        return lcs[m][n]
    
    @staticmethod
    def compute(reference, hypothesis):
        """
        Compute ROUGE-L score (F-measure of LCS)
        Args:
            reference: Reference string
            hypothesis: Hypothesis string
        Returns:
            rouge_l: F-measure based on LCS
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        lcs_len = RougeScore._lcs_length(ref_tokens, hyp_tokens)
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0
        
        recall = lcs_len / len(ref_tokens)
        precision = lcs_len / len(hyp_tokens)
        
        if recall + precision == 0:
            f_measure = 0.0
        else:
            f_measure = 2 * (recall * precision) / (recall + precision)
        
        return f_measure


class SemanticSimilarity:
    """Compute word overlap similarity"""
    
    @staticmethod
    def compute(reference, hypothesis):
        """
        Compute Jaccard similarity (word overlap)
        """
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if len(ref_words | hyp_words) == 0:
            return 0.0
        
        intersection = len(ref_words & hyp_words)
        union = len(ref_words | hyp_words)
        
        return intersection / union


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_results(results_path=None):
    """Evaluate generated conclusions"""
    
    print("\n" + "="*70)
    print("EVALUATION PIPELINE: ARGUMENT GENERATION METRICS")
    print("="*70)
    
    # Load results
    if results_path is None:
        results_path = Path('/Users/poures/Desktop/PC/image-arg/inference_results/generated_conclusions.json')
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n[1/2] Evaluating {len(results)} samples...\n")
    
    bleu_scores = []
    rouge_scores = []
    similarity_scores = []
    confidences = []
    
    eval_results = []
    
    for item in results:
        ground_truth = item['ground_truth']
        generated = item['generated']
        
        # Compute metrics
        bleu = BleuScore.compute(ground_truth, generated)
        rouge = RougeScore.compute(ground_truth, generated)
        similarity = SemanticSimilarity.compute(ground_truth, generated)
        confidence = item['confidence']
        
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        similarity_scores.append(similarity)
        confidences.append(confidence)
        
        eval_results.append({
            'index': item['index'],
            'premise': item['premise'],
            'ground_truth': ground_truth,
            'generated': generated,
            'confidence': confidence,
            'bleu': bleu,
            'rouge': rouge,
            'semantic_similarity': similarity
        })
        
        print(f"Sample {item['index']}:")
        print(f"  BLEU:  {bleu:.4f}")
        print(f"  ROUGE: {rouge:.4f}")
        print(f"  Semantic Sim: {similarity:.4f}")
        print(f"  Model Confidence: {confidence:.4f}\n")
    
    # Aggregate metrics
    print("="*70)
    print("AGGREGATE METRICS:")
    print("="*70)
    
    avg_bleu = np.mean(bleu_scores)
    avg_rouge = np.mean(rouge_scores)
    avg_similarity = np.mean(similarity_scores)
    avg_confidence = np.mean(confidences)
    
    metrics = {
        'num_samples': len(results),
        'bleu': {
            'mean': avg_bleu,
            'std': np.std(bleu_scores),
            'min': np.min(bleu_scores),
            'max': np.max(bleu_scores)
        },
        'rouge': {
            'mean': avg_rouge,
            'std': np.std(rouge_scores),
            'min': np.min(rouge_scores),
            'max': np.max(rouge_scores)
        },
        'semantic_similarity': {
            'mean': avg_similarity,
            'std': np.std(similarity_scores),
            'min': np.min(similarity_scores),
            'max': np.max(similarity_scores)
        },
        'model_confidence': {
            'mean': avg_confidence,
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
    }
    
    print(f"\nBLEU Score (1-4 grams):")
    print(f"  Mean:   {avg_bleu:.4f}")
    print(f"  Std:    {np.std(bleu_scores):.4f}")
    print(f"  Range:  [{np.min(bleu_scores):.4f}, {np.max(bleu_scores):.4f}]")
    
    print(f"\nROUGE Score (LCS F-measure):")
    print(f"  Mean:   {avg_rouge:.4f}")
    print(f"  Std:    {np.std(rouge_scores):.4f}")
    print(f"  Range:  [{np.min(rouge_scores):.4f}, {np.max(rouge_scores):.4f}]")
    
    print(f"\nSemantic Similarity (Jaccard):")
    print(f"  Mean:   {avg_similarity:.4f}")
    print(f"  Std:    {np.std(similarity_scores):.4f}")
    print(f"  Range:  [{np.min(similarity_scores):.4f}, {np.max(similarity_scores):.4f}]")
    
    print(f"\nModel Confidence:")
    print(f"  Mean:   {avg_confidence:.4f}")
    print(f"  Std:    {np.std(confidences):.4f}")
    print(f"  Range:  [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
    
    # Save results
    print(f"\n[2/2] Saving evaluation results...")
    
    output_dir = Path('/Users/poures/Desktop/PC/image-arg/inference_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✓ Metrics saved to: {output_dir / 'evaluation_metrics.json'}")
    print(f"✓ Details saved to: {output_dir / 'detailed_results.json'}")
    
    return metrics, eval_results


if __name__ == '__main__':
    metrics, eval_results = evaluate_results()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nModel Summary:")
    print(f"  Average BLEU: {metrics['bleu']['mean']:.4f}")
    print(f"  Average ROUGE: {metrics['rouge']['mean']:.4f}")
    print(f"  Average Semantic Similarity: {metrics['semantic_similarity']['mean']:.4f}")
