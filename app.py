from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

true_dataset_path = 'True.csv'
df_true = pd.read_csv(true_dataset_path)
fake_dataset_path = 'Fake.csv'
df_fake = pd.read_csv(fake_dataset_path)
df = pd.concat([df_true, df_fake], ignore_index=True)

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(df['text'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_results = search_news(query)
    return render_template('search_results.html', query=query, results=search_results)

def search_news(query, top_k=5):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, text_vectors).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_k]
    top_news = df.iloc[top_indices]['text']
    return top_news

if __name__ == '__main__':
    app.run(debug=True)
