"""
Text utility functions for preprocessing and analysis.
"""
import re
from typing import Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special unicode characters
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\u200c', '')
    text = text.replace('\u200d', '')
    
    # Normalize excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()


def get_text_statistics(text: str) -> Dict[str, float]:
    """
    Get statistical features of text.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    }


