import os
import re
import pickle
from gensim.models import CoherenceModel
import numpy as np

OUTPUT_DIR = "output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

def save_cache(obj, name):

    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  ✓ Cached: {name}")


def load_cache(name):

    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"  ✓ Loaded from cache: {name}")
            return pickle.load(f)
    return None


def save_embeddings_cache(embeddings, name):

    path = os.path.join(CACHE_DIR, f"{name}.npy")
    np.save(path, embeddings)
    print(f"  ✓ Cached embeddings: {name}")


def load_embeddings_cache(name):

    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        print(f"  ✓ Loaded embeddings from cache: {name}")
        return np.load(path)
    return None


def cache_exists(name, is_embedding=False):

    ext = ".npy" if is_embedding else ".pkl"
    path = os.path.join(CACHE_DIR, f"{name}{ext}")
    return os.path.exists(path)



def clean_text_minimal(text, label_to_remove=None):

    if not isinstance(text, str):
        return ""
    
    text = text.lower()


    if label_to_remove:
        label_to_remove = label_to_remove.lower()
        pattern = re.compile(re.escape(label_to_remove), re.IGNORECASE)
        text = pattern.sub('', text)
    

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text_for_topics(text, stopwords_set):

    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zàèéìòù\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [w for w in words if w not in stopwords_set and len(w) > 2]
    
    return ' '.join(words)


def tokenize_for_gensim(texts):
    return [text.split() for text in texts if text.strip()]


def calculate_coherence(model, corpus, dictionary, texts, coherence_type='c_v'):

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

    try:
        return model.log_perplexity(corpus)
    except Exception:
        return None


def get_topic_words(model, n_words=10):

    topics = []
    for topic_id in range(model.num_topics):
        topic_terms = model.show_topic(topic_id, n_words)
        words = [word for word, _ in topic_terms]
        topics.append(words)
    return topics


def calculate_topic_diversity(topics, n_top_words=10):

    all_words = []
    for topic in topics:
        all_words.extend(topic[:n_top_words])
    unique_words = len(set(all_words))
    total_words = len(all_words)
    return unique_words / total_words if total_words > 0 else 0