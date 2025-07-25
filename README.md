# 🎬 Movie Recommendation System

A content-based movie recommender system using TF-IDF and cosine similarity. Built with Python and real-world data from TMDB.

## 🔍 How It Works

- Combines keywords, genres, cast, director, and more into a unified "tag"
- Uses TF-IDF vectorization and cosine similarity to recommend similar movies
- Intelligent fuzzy matching helps even with typos

## 📦 Example

```python
from src.recommender import MovieRecommendationSystem

recommender = MovieRecommendationSystem("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv")
recommendations = recommender.recommend("Avatar")

for movie in recommendations:
    print(movie)
```

## 📁 Files

- `src/recommender.py` — main class
- `notebook/demo.ipynb` — demo notebook
- `data/` — contains dataset CSVs (or provide a link instead)
- `requirements.txt` — required libraries

## 🧪 Sample Output

```
{'title': 'Aliens', 'rating': 7.9, 'popularity': 22.0, 'similarity_score': 0.212}
{'title': 'Titanic', 'rating': 7.8, 'popularity': 45.1, 'similarity_score': 0.198}
```

## 📌 Requirements

- Python 3.8+
- pandas
- scikit-learn

Install with:

```bash
pip install -r requirements.txt
```

## 🚀 Try It!

Run the Jupyter notebook in `/notebook/demo.ipynb` or integrate the system in your own projects.

## 📖 License

MIT
