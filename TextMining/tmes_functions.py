import os
import re
import pickle
from gensim.models import CoherenceModel
import numpy as np
# ==============================================================================
# CACHING FUNCTIONS
# ==============================================================================
OUTPUT_DIR = "output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

def save_cache(obj, name):
    """Save object to cache using pickle."""
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  ✓ Cached: {name}")


def load_cache(name):
    """Load object from cache if exists, otherwise return None."""
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"  ✓ Loaded from cache: {name}")
            return pickle.load(f)
    return None


def save_embeddings_cache(embeddings, name):
    """Save embeddings as numpy file for efficient storage."""
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    np.save(path, embeddings)
    print(f"  ✓ Cached embeddings: {name}")


def load_embeddings_cache(name):
    """Load embeddings from numpy file if exists."""
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        print(f"  ✓ Loaded embeddings from cache: {name}")
        return np.load(path)
    return None


def cache_exists(name, is_embedding=False):
    """Check if cache file exists."""
    ext = ".npy" if is_embedding else ".pkl"
    path = os.path.join(CACHE_DIR, f"{name}{ext}")
    return os.path.exists(path)

# ==============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ==============================================================================

def clean_text_minimal(text, label_to_remove=None):
    """
    Minimal cleaning for contextual embeddings with DATA LEAKAGE PREVENTION.
    
    Rationale: Contextual embedding models like BERT preserve semantic meaning
    through subword tokenization and context. Aggressive preprocessing (stemming,
    lemmatization, stopword removal) destroys this context and degrades performance.
    
    Args:
        text: The job description text
        label_to_remove: The job category/title to remove from text (leakage prevention)
    
    Returns:
        Minimally cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # LEAKAGE PREVENTION: Remove the category name from the text
    # This prevents the model from "cheating" by finding the label in the description
    if label_to_remove:
        label_to_remove = label_to_remove.lower()
        pattern = re.compile(re.escape(label_to_remove), re.IGNORECASE)
        text = pattern.sub('', text)
    
    # Basic whitespace normalization only
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text_for_topics(text, stopwords_set):
    """
    Aggressive cleaning for traditional topic modeling (LDA, HDP, LSA).
    
    Rationale: Bag-of-words models count term frequencies. Common words (stopwords)
    dominate counts and obscure meaningful topic terms. Aggressive filtering is
    essential for interpretable topics.
    
    Args:
        text: The job description text
        stopwords_set: Set of stopwords to remove
    
    Returns:
        Cleaned text with only content words
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zàèéìòù\s]', ' ', text)  # Keep only letters (including Italian accents)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and filter
    words = text.split()
    words = [w for w in words if w not in stopwords_set and len(w) > 2]
    
    return ' '.join(words)


def tokenize_for_gensim(texts):
    """Convert cleaned texts to list of word lists for gensim."""
    return [text.split() for text in texts if text.strip()]

# ==============================================================================
# TOPIC MODELING METRICS
# ==============================================================================

def calculate_coherence(model, corpus, dictionary, texts, coherence_type='c_v'):
    """
    Calculate coherence score for a gensim model.
    
    Coherence measures how semantically related the top words of each topic are.
    Higher coherence = more interpretable topics.
    """
    try:
        coherence_model = CoherenceModel(
            model=model, texts=texts, corpus=corpus,
            dictionary=dictionary, coherence=coherence_type
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"  Warning: Coherence calculation failed: {e}")
        return None


def calculate_perplexity(model, corpus):
    """
    Calculate perplexity for a gensim LDA model.
    
    Perplexity measures how well the model predicts held-out data.
    Lower perplexity = better generalization.
    """
    try:
        return model.log_perplexity(corpus)
    except Exception:
        return None


def get_topic_words(model, n_words=10):
    """Extract top words for each topic from a gensim model."""
    topics = []
    for topic_id in range(model.num_topics):
        topic_terms = model.show_topic(topic_id, n_words)
        words = [word for word, _ in topic_terms]
        topics.append(words)
    return topics


def calculate_topic_diversity(topics, n_top_words=10):
    """
    Calculate topic diversity (proportion of unique words across topics).
    
    Higher diversity = topics cover different vocabulary.
    Low diversity = topics share many words (redundant).
    """
    all_words = []
    for topic in topics:
        all_words.extend(topic[:n_top_words])
    unique_words = len(set(all_words))
    total_words = len(all_words)
    return unique_words / total_words if total_words > 0 else 0