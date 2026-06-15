from flask import Flask, request, jsonify, render_template, session
import uuid

from books_recommender import initialize_books, initiate_RAG_pipeline
from flask_cors import CORS
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, resources={r"/*": {"origins": "*"}})
sessions = {}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of books_recommender.py
books_data = initialize_books()
qa_chain, vector_store = initiate_RAG_pipeline(books_data)
conversation_state = {}
@app.route("/")
def home():
    session.clear()
    return render_template("chat.html")


@app.route('/start', methods=['GET'])
def start():
    """Start a new conversation"""
    session_id = str(uuid.uuid4())
    conversation_state[session_id] = {
        'step': 'search',
        'history': []
    }

    message = "Welcome! What book are you looking for?"
    conversation_state[session_id]['history'].append({"role": "assistant", "content": message})

    return jsonify({
        "session_id": session_id,
        "message": message
    })


@app.route('/chat', methods=['POST'])
def chat():
    """Process user query through RAG pipeline."""
    data = request.json
    session_id = data.get('session_id')
    user_input = data.get('message', '').strip()

    if session_id not in conversation_state:
        return jsonify({"error": "Invalid session"}), 400

    if not user_input:
        return jsonify({"message": "Please enter a search query"}), 400

    state = conversation_state[session_id]
    state['history'].append({"role": "user", "content": user_input})

    try:
        # Process through RAG chain
        result = qa_chain.invoke({"query": user_input})

        response = result['result']
        source_docs = result.get('source_documents', [])
        print(source_docs[0])
        print("\n🔍 DEBUG: Result keys:", result.keys())
        print(f"Source docs count: {len(source_docs)}")

        # Format recommendations
        retrieved_docs = vector_store.similarity_search(user_input, k=5)
        print("\n🔍 DEBUG: Docs:", retrieved_docs)
        recommendations = []
        for doc in retrieved_docs[:5]:
            recommendations.append({
                "title": doc.metadata.get('title', 'Unknown'),
                "author": doc.metadata.get('author', 'Unknown'),
                "genre": doc.metadata.get('genres', 'Unknown'),
                "content": doc.metadata.get('description', 'Unknown'),
            })


        state['history'].append({"role": "assistant", "content": response})

        return jsonify({
            "message": response,
            "recommendations": recommendations,
            "ask_continue": True
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/continue', methods=['POST'])
def continue_chat():
    """Handle continuation or new search."""
    data = request.json
    user_input = data.get('message', '').strip().lower()

    if any(word in user_input for word in ['yes', 'yeah', 'sure', 'another', 'more']):
        return jsonify({
            "message": "What else would you like to search for?",
            "continue_search": True
        })
    else:
        return jsonify({
            "message": "Thank you for using our book recommendation chatbot! 📚",
            "end": True
        })


if __name__ == "__main__":
    app.run(debug=True)

