# Book Recommendation Chatbot

A simple chatbot web app built with Flask that recommends books based on user preferences (by *genre* or *author*). The chatbot maintains a short conversation flow and suggests books from a dataset.


## 📂 Project Structure
```
.
├── Books_recommender.ipynb
├── LICENSE
├── README.md
├── __pycache__
├── app.py
├── assets
│   ├── books_1.Best_Books_Ever.csv
│   ├── books_modified.csv
│   └── glove.twitter.27B.50d.txt
├── books_recommender.py
├── requirements.txt
├── static
└── templates
    └── chat.html
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/GraceCeline/BookChatbot2.git
cd book-chatbot

```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
python src/app.py
```

4. Open in browser

   Go to 👉 http://127.0.0.1:5000

```
to run the program in the terminal. Note that the program in the terminal is slightly different than Flask app.

