import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the query classifier
zero_shot_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
query_labels = ["chitchat", "wiki_query"]
query_hypothesis_template = "This is a {}."

# Load the chitchat model
blenderbot_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Load the T5 model for summarization
t5_model_name = "t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Reload the zero-shot classification model and tokenizer for topic classification
load_directory = "./saved_zero_shot_model"
tokenizer = AutoTokenizer.from_pretrained(load_directory)
model = AutoModelForSequenceClassification.from_pretrained(load_directory)
pipe = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define topics and hypothesis template for topic classification
topics = [
    "Health",
    "Environment",
    "Technology",
    "Economy",
    "Entertainment",
    "Sports",
    "Politics",
    "Education",
    "Travel",
    "Food",
]
topic_hypothesis_template = "This query is related to {}."

# Functions
def classify_query_type(query, classifier, labels, threshold=0.5):
    """
    Classifies the query as either 'chitchat' or 'wiki_query'.
    """
    output = classifier(query, labels, hypothesis_template=query_hypothesis_template)
    scores = output["scores"]
    best_label = output["labels"][0]
    best_score = scores[0]
    return {"label": best_label, "score": best_score} if best_score > threshold else {"label": "uncertain", "score": best_score}

def chat_with_blenderbot(input_text):
    """
    Handles chitchat using BlenderBot.
    """
    inputs = blenderbot_tokenizer(input_text, return_tensors="pt")
    outputs = blenderbot_model.generate(**inputs)
    return blenderbot_tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_multi_topics(query, pipe, topics, threshold=0.1, top_n=5):
    """
    Classifies the query into multiple topics using zero-shot classification.
    """
    output = pipe(query, topics, hypothesis_template=topic_hypothesis_template)
    labels = output["labels"]
    scores = output["scores"]
    return [label for label, score in zip(labels, scores) if score > threshold][:top_n]

# Preprocess text function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [
        lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words
    ]
    return " ".join(filtered_words)

# Load preprocessed data function
def load_preprocessed_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Create a TF-IDF vectorizer function
def create_tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

# Retrieve most relevant articles with unique URLs
def get_most_relevant_articles(query, tfidf_vectorizer, tfidf_matrix, articles, top_n=3):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]

    unique_articles = []
    seen_urls = set()

    for idx in sorted_indices:
        if len(unique_articles) >= top_n:
            break
        article = articles[idx]
        article_url = article.get('url', None)
        if article_url and article_url not in seen_urls:
            unique_articles.append(article)
            seen_urls.add(article_url)

    return unique_articles

# Generate a meaningful summary using T5
def generate_meaningful_summary_t5(combined_text, max_length=800, min_length=100):
    inputs = t5_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = t5_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Combine summaries of the most relevant articles
def combine_summaries(articles):
    combined_text = " ".join([
        article['summary']['text_en'] if 'summary' in article and 'text_en' in article['summary'] else "No Summary"
        for article in articles
    ])
    return generate_meaningful_summary_t5(combined_text)

# Main QA function
def wiki_qa_system(query, preprocessed_data, top_n=3):  # Set top_n to 5
    """
    Main Wiki QA system function.
    """
    relevant_topics_data = classify_multi_topics(query, pipe, topics, threshold=0.1, top_n=5)
    relevant_topics = [t["topic"] for t in relevant_topics_data]
    print(f"Relevant Topics: {relevant_topics}")

    all_articles = []
    for topic, data in preprocessed_data.items():
        if topic in relevant_topics and 'articles' in data and isinstance(data['articles'], list):
            for article in data['articles']:
                if isinstance(article, dict) and 'preprocessed_summary' in article:
                    article['topic'] = topic
                    all_articles.append(article)

    if not all_articles:
        raise ValueError("No valid articles found for the relevant topics.")

    articles_summaries = [
        preprocess_text(article['preprocessed_summary']) for article in all_articles
        if 'preprocessed_summary' in article
    ]

    if not articles_summaries:
        raise ValueError("No preprocessed summaries found for matching.")

    tfidf_vectorizer, tfidf_matrix = create_tfidf_vectorizer(articles_summaries)
    most_relevant_articles = get_most_relevant_articles(query, tfidf_vectorizer, tfidf_matrix, all_articles, top_n)
    combined_summary = combine_summaries(most_relevant_articles)

    answers = [{
        'title': article.get('title', "No Title"),
        'topic': article.get('topic', "No Topic"),
        'url': article.get('url', "No URL")
    } for article in most_relevant_articles]

    return {
        'combined_summary': combined_summary,
        'answers': answers
    }

# Integration of systems
if __name__ == "__main__":
    preprocessed_filename = 'preprocessed_data.json'
    try:
        # Load preprocessed data for wiki QA
        preprocessed_data = load_preprocessed_data(preprocessed_filename)
        print("Preprocessed data loaded successfully.")
        
        print("Enter your query below (type 'exit' to quit):")
        while True:
            query = input("Enter query: ")
            if query.lower() == "exit":
                print("Exiting...")
                break
            
            # Classify the query type
            query_type = classify_query_type(query, zero_shot_classifier, query_labels, threshold=0.5)
            if query_type["label"] == "chitchat":
                # Handle chitchat
                response = chat_with_blenderbot(query)
                print(f"BlenderBot: {response}")
            elif query_type["label"] == "wiki_query":
                # Handle wiki query
                result = wiki_qa_system(query, preprocessed_data, top_n=3)
                print("\nSummary:")
                print(result['combined_summary'])
                print("\nRelevant Articles:")
                for answer in result['answers']:
                    print(f"Title: {answer['title']}")
                    print(f"Topic: {answer['topic']}")
                    print(f"URL: {answer['url']}")
                    print()
            else:
                print("The query could not be classified confidently. Please try again.")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Integration of systems
if __name__ == "__main__":
    preprocessed_filename = 'preprocessed_data.json'
    try:
        # Load preprocessed data for wiki QA
        preprocessed_data = load_preprocessed_data(preprocessed_filename)
        print("Preprocessed data loaded successfully.")
        
        print("Enter your query below (type 'exit' to quit):")
        while True:
            query = input("Enter query: ")
            if query.lower() == "exit":
                print("Exiting...")
                break
            
            # Classify the query type
            query_type = classify_query_type(query, zero_shot_classifier, query_labels, threshold=0.5)
            if query_type["label"] == "chitchat":
                # Handle chitchat
                response = chat_with_blenderbot(query)
                print(f"BlenderBot: {response}")
            elif query_type["label"] == "wiki_query":
                # Handle wiki query
                result = wiki_qa_system(query, preprocessed_data, top_n=5)
                print("\nSummary:")
                print(result["summary"])
                print("\nRelevant Articles:")
                for answer in result["answers"]:
                    print(f"Title: {answer['title']}")
                    print(f"Topic: {answer['topic']}")
                    print(f"URL: {answer['url']}")
                    print()
            else:
                print("The query could not be classified confidently. Please try again.")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




