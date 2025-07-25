import streamlit as st
import pandas as pd
from src.recommender import MovieRecommendationSystem

# App config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# Title
st.title("🎬 Movie Recommendation System")
st.markdown("Get content-based recommendations based on your favorite movie!")

# Initialize recommender (only once)
@st.cache_resource
def load_recommender():
    recommender = MovieRecommendationSystem(
        "data/tmdb_5000_movies.csv",
        "data/tmdb_5000_credits.csv"
    )
    recommender.load_and_preprocess_data()
    recommender.build_similarity_matrix()
    return recommender

recommender = load_recommender()

# Input box
movie_input = st.text_input("Enter a movie title:", value="Inception")

# Number of recommendations
top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

# Recommend button
if st.button("🎯 Recommend Movies"):
    with st.spinner("Finding similar movies..."):
        result = recommender.recommend(movie_input, num_recommendations=top_n)

        if "error" in result:
            st.error(result["error"])
            if "suggestions" in result:
                st.info("Did you mean: " + ", ".join(result["suggestions"]))
        else:
            st.success(f"Based on: {result['input_movie']}")
            for i, rec in enumerate(result["recommendations"], 1):
                st.markdown(f"""
                    **{i}. {rec['title']}**
                    - ⭐ Rating: {rec['rating']}
                    - 🔥 Popularity: {rec['popularity']}
                    - 🧠 Similarity Score: {rec['similarity_score']}
                """)