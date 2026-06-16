import pandas as pd
import numpy as np
import ast
# from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from pathlib import Path
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
import re
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
glove_file = os.path.join(BASE_DIR.parent, "assets", "glove.twitter.27B.50d.txt")
# model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
print(CHROMA_DB_PATH)


def get_preference():
  category = input("Do you want to look for recommendation according to author or genre? ")
  return category

def get_genres():
  genre = input("What kind of book genre do you like? If you want to put in 2 or more genres please separate them with a comma. ")
  return genre

def genre_match(gs, genre_input):
    return all(g.lower() in (x.lower() for x in gs) for g in genre_input)

def get_author():
  author = input("Who is the author you want to look for? ")
  return author

def get_keywords():
  keywords = input("What are the keywords you want to look for? ")
  return keywords.split()


def show_books(book_list):
        for idx, row in book_list.iterrows():
            print(f"{row['title']} by {row['author']} (Genres: {', '.join(row['genres'])})")

def get_top_keywords(row_idx, tfidf_mat, features, top_n=5):
    row = tfidf_mat.getrow(row_idx)
    nonzero_idx = row.nonzero()[1]
    scores = row.data
    top_indices = scores.argsort()[::-1][:top_n]
    return features[nonzero_idx][top_indices].tolist()

"""
def extract_tfidf_keywords(books_data):
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]+\b', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_data['description'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    books_data['keywords'] = [get_top_keywords(i, tfidf_matrix, feature_names) for i in range(len(books_data))]
    keybert_keywords['keywords1'] = keybert_keywords['keywords1'].apply(lambda keywords: keywords.split())
    books_data['keywords_comb'] = [
        list(set(k1 + k2))
        for k1, k2 in zip(books_data['keywords'], keybert_keywords['keywords1'])
    ]
    books_data['keywords'] = books_data['keywords_comb']
    return books_data
"""

"""def words_to_vector(words, model=model):
    vectors = [model.get_vector(word) if word in model else np.zeros(model.vector_size) for word in words]
    return np.mean(vectors, axis=0)"""


"""def add_book_vectors(books_data):
    books_data['vector'] = books_data['keywords'].apply(words_to_vector)
    return books_data"""

def remove_non_alphabetic(desc):
    try:
        return re.sub(r"[^a-zA-Z\s]+", "", desc)
    except:
        return desc

# Preprocess dataset
def initialize_books():
    books_file = os.path.join(BASE_DIR.parent, "assets", "books_1.Best_Books_Ever.csv")
    books_data = pd.read_csv(books_file,
                             usecols=['title', 'author', 'rating', 'numRatings', 'description', 'language', 'genres'])
    books_data = books_data.reset_index(drop=True)
    books_data = books_data.drop_duplicates(subset=['title'])
    books_data["length"] = books_data['description'].apply(lambda d: len(d.split()) if isinstance(d, str) else 0)
    books_data = books_data[books_data["length"] >= 4].copy()
    books_data.dropna(subset=["description"], inplace=True)
    books_data["description"] = books_data["description"].apply(remove_non_alphabetic)
    books_data['genres'] = books_data['genres'].apply(ast.literal_eval)
    books_data['author'] = books_data['author'].apply(lambda x: x.lower())

    # Rating score
    books_data['rating_total'] = books_data['rating'] * books_data['numRatings']

    return books_data


def build_documents(books_df):
    """Convert DataFrame to LangChain Documents."""
    documents = []
    for idx, row in books_df.iterrows():
        # Handle genres - convert to list and filter empty values
        genres = row['genres']

        # Handle NaN/None
        if pd.isna(genres).all():
            genres_list = ['Unknown']
        elif isinstance(genres, str):
            # Split by comma and strip whitespace, filter empty strings
            genres_list = [g.strip() for g in genres.split(',') if g.strip()]
            # If empty after filtering, add default
            if not genres_list:
                genres_list = ['Unknown']
        elif isinstance(genres, list):
            # Already a list, just ensure it's not empty
            genres_list = genres if genres else ['Unknown']
        else:
            genres_list = ['Unknown']

        # Handle other fields similarly
        title = str(row['title']) if not pd.isna(row['title']) else 'Unknown'
        author = str(row['author']) if not pd.isna(row['author']) else 'Unknown'
        description = str(row['description']) if not pd.isna(row['description']) else 'No description'

        doc = Document(
            page_content=description,
            metadata={
                'title': title,
                'author': author,
                'genres': genres_list,  # Always a non-empty list
                'rating': float(row.get('rating', 0)) if not pd.isna(row.get('rating')) else 0,
                'language': str(row.get('language', 'Unknown')) if not pd.isna(row.get('language')) else 'Unknown'
            }
        )
        documents.append(doc)

    print(f"✅ Built {len(documents)} documents")
    return documents

def initiate_RAG_pipeline(books_df):
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    if os.path.exists(CHROMA_DB_PATH):
        print("✅ Loading Chroma database from disk...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        print("✅ Chroma database loaded!")
    else:
        book_docs = build_documents(books_df)
        # Split into chunks
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        books = text_splitter.split_documents(book_docs)

        print(f"✅ Split into {len(books)} chunks")

        vector_store = Chroma.from_documents(
                documents=books[:100],
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )
        print(f"✅ Chroma database created and saved to {CHROMA_DB_PATH}")

    # Create RAG chain with RetrievalQA
    mistral = ChatMistralAI(model_name="mistral-small", temperature=0.0, max_tokens=1024)

    qa_chain = RetrievalQA.from_chain_type(
        llm=mistral,
        chain_type="stuff",  # "stuff" = combine all docs into one prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=True  # Add verbose to see what's happening  # Get the source docs back
    )

    return qa_chain, vector_store


