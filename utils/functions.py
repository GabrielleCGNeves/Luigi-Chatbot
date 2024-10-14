import random
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text, nlp, stop_words):
    """
    Preprocess the text by removing URLs, extra spaces, punctuation, and stop words.
    Performs lemmatization on the tokens.

    Args:
        text (str): Text to be preprocessed.
        nlp (callable): Natural language processing function.
        stop_words (set): Set of stop words to be removed.

    Returns:
        str: Preprocessed text.
    """
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r" +", ' ', text)

    document = nlp(text)
    tokens = [token.lemma_ for token in document if token.lemma_ not in stop_words and token.lemma_ not in string.punctuation]
    tokens = ' '.join([str(token) for token in tokens if not token.isdigit()])

    return tokens


def greet_user(text):
    """
    Responds to a greeting with a random greeting.

    Args:
        text (str): User input text.

    Returns:
        str: Greeting response.
    """
    greeting_inputs = {'bom dia', 'salve', 'oi', 'eae', 'olá', 'coé'}
    greeting_responses = ['hey', 'olá', 'opa', 'oiee']

    for word in text.lower().split():
        if word in greeting_inputs:
            return random.choice(greeting_responses)
    return None


def generate_response(user_text, sentence_list, preprocessed_sentence_list):
    """
    Generates a response based on the similarity of the user's text with known sentences.

    Args:
        user_text (str): User input text.
        sentence_list (list): List of known sentences.
        preprocessed_sentence_list (list): List of preprocessed sentences.

    Returns:
        str: Chatbot response.
    """
    preprocessed_sentence_list.append(user_text)

    tfidf = TfidfVectorizer()
    tfidf_vectorized_words = tfidf.fit_transform(preprocessed_sentence_list)

    similarity_scores = cosine_similarity(tfidf_vectorized_words[-1], tfidf_vectorized_words)
    selected_sentence_index = similarity_scores.argsort()[0][-2]
    similarity_vector = similarity_scores.flatten()
    similarity_vector.sort()
    found_vector = similarity_vector[-2]

    if found_vector == 0:
        return 'Desculpe, não entendi! :/'
    else:
        return sentence_list[selected_sentence_index]