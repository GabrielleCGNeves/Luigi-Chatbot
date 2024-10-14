import urllib.request
import bs4 as bs
import nltk
import spacy
from utils.functions import generate_response, greet_user, preprocess_text
import os

nltk.download("punkt")
spacy.cli.download("pt_core_news_sm")

os.system('cls' if os.name == 'nt' else 'clear')



def fetch_wikipedia_content(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = bs.BeautifulSoup(html, "lxml")
    paragraphs = soup.find_all("p")
    content = " ".join([p.text for p in paragraphs])
    return content.lower()


def preprocess_sentences(sentences, nlp, stop_words):
    return [preprocess_text(sentence, nlp, stop_words) for sentence in sentences]


def main():
    url = "https://pt.wikipedia.org/wiki/Super_Mario"
    content = fetch_wikipedia_content(url)
    sentences = nltk.sent_tokenize(content)

    nlp = spacy.load("pt_core_news_sm")
    stop_words = spacy.lang.pt.stop_words.STOP_WORDS

    preprocessed_sentences = preprocess_sentences(sentences, nlp, stop_words)

    print("Olá, me chamo Luigi e vou responder perguntas sobre o Super Mário: ")
    while True:
        user_input = input("Usuario: ").lower()
        if user_input == "sair":
            print("Luigi: Até breve!")
            break
        if greet_user(user_input):
            print("Luigi: " + greet_user(user_input) + "\n")
        else:
            response = generate_response(preprocess_text(user_input, nlp, stop_words), sentences, preprocessed_sentences)
            print("Luigi: " + response + "\n")
            preprocessed_sentences.remove(preprocess_text(user_input, nlp, stop_words))


if __name__ == "__main__":
    main()
