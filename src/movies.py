import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

class MovieRecommendationSystem:
    def __init__(self, movies_path, credits_path):
        """Initialize the recommendation system with data paths."""
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = None
        self.cleaned_df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the movie data."""
        print("Loading data...")
        
        # Load datasets
        self.movies_df = pd.read_csv(self.movies_path)
        credits_df = pd.read_csv(self.credits_path)
        
        # Merge datasets on movie_id (assuming both have this column)
        # If using title instead, change to: on='title'
        if 'movie_id' in self.movies_df.columns and 'movie_id' in credits_df.columns:
            merged_df = self.movies_df.merge(credits_df, on='movie_id', how='left')
        else:
            # Fallback to title-based merge
            merged_df = self.movies_df.merge(credits_df, on='title', how='left')
        
        print(f"Loaded {len(merged_df)} movies")
        
        # Create comprehensive tags for each movie
        merged_df['tags'] = merged_df.apply(self._create_movie_tags, axis=1)
        
        # Clean and prepare final dataset
        self.cleaned_df = merged_df[['title', 'tags', 'vote_average', 'popularity']].dropna()
        self.cleaned_df['tags'] = self.cleaned_df['tags'].apply(self._preprocess_text)
        
        print(f"Preprocessed {len(self.cleaned_df)} movies")
        return self.cleaned_df
    
    def _extract_names(self, text_list, key='name', max_items=3):
        """Extract names from JSON-like string format."""
        try:
            if pd.isna(text_list):
                return []
            
            # Handle string representation of list
            if isinstance(text_list, str):
                items = ast.literal_eval(text_list)
            else:
                items = text_list
                
            if isinstance(items, list):
                return [item[key] for item in items[:max_items] if isinstance(item, dict) and key in item]
            return []
        except:
            return []
    
    def _create_movie_tags(self, row):
        """Create comprehensive tags for a movie combining multiple features."""
        tags = []
        
        # Add overview/plot
        if pd.notna(row.get('overview')):
            tags.append(str(row['overview']))
        
        # Add genres
        genres = self._extract_names(row.get('genres', []))
        tags.extend(genres)
        
        # Add keywords
        keywords = self._extract_names(row.get('keywords', []))
        tags.extend(keywords)
        
        # Add cast (top 3)
        cast = self._extract_names(row.get('cast', []))
        tags.extend(cast)
        
        # Add crew (director, producer)
        crew = self._extract_names(row.get('crew', []))
        for person in crew:
            if any(job in str(person).lower() for job in ['director', 'producer']):
                tags.append(person)
        
        # Add production companies
        companies = self._extract_names(row.get('production_companies', []))
        tags.extend(companies[:2])  # Top 2 companies
        
        return ' '.join(tags)
    
    def _preprocess_text(self, text):
        """Clean and preprocess text data."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def build_similarity_matrix(self):
        """Build TF-IDF matrix and compute cosine similarity."""
        print("Building similarity matrix...")
        
        # Initialize TF-IDF Vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,        # Limit vocabulary size
            ngram_range=(1, 2),        # Include unigrams and bigrams
            min_df=2,                  # Ignore rare terms
            max_df=0.8,                # Ignore too common terms
            lowercase=True,
            strip_accents='ascii'
        )
        
        # Fit and transform the tags
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_df['tags'])
        
        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print("Similarity matrix built successfully!")
        return self.similarity_matrix
    
    def find_movie_index(self, movie_title):
        """Find movie index with fuzzy matching."""
        movie_title = movie_title.lower().strip()
        titles = self.cleaned_df['title'].str.lower().str.strip().tolist()
        
        # Exact match
        for i, title in enumerate(titles):
            if title == movie_title:
                return i
        
        # Fuzzy match
        close_matches = get_close_matches(movie_title, titles, n=1, cutoff=0.6)
        if close_matches:
            return titles.index(close_matches[0])
        
        return None
    
    def get_recommendations(self, movie_title, num_recommendations=5, include_scores=True):
        """
        Get movie recommendations based on content similarity.
        
        Args:
            movie_title (str): Title of the movie to base recommendations on
            num_recommendations (int): Number of recommendations to return
            include_scores (bool): Whether to include similarity scores
            
        Returns:
            list: List of recommended movies with optional similarity scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not built. Call build_similarity_matrix() first.")
        
        # Find movie index
        movie_idx = self.find_movie_index(movie_title)
        
        if movie_idx is None:
            # Suggest similar titles
            titles = self.cleaned_df['title'].str.lower().tolist()
            suggestions = get_close_matches(movie_title.lower(), titles, n=3, cutoff=0.3)
            return {
                'error': f"Movie '{movie_title}' not found.",
                'suggestions': [self.cleaned_df.iloc[titles.index(s)]['title'] for s in suggestions]
            }
        
        # Get similarity scores for this movie
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity (excluding the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        # Prepare recommendations
        recommendations = []
        for idx, score in sim_scores:
            movie_info = {
                'title': self.cleaned_df.iloc[idx]['title'],
                'rating': round(self.cleaned_df.iloc[idx]['vote_average'], 1),
                'popularity': round(self.cleaned_df.iloc[idx]['popularity'], 1)
            }
            
            if include_scores:
                movie_info['similarity_score'] = round(score, 3)
            
            recommendations.append(movie_info)
        
        return {
            'input_movie': self.cleaned_df.iloc[movie_idx]['title'],
            'recommendations': recommendations
        }
    
    def get_movie_info(self, movie_title):
        """Get detailed information about a specific movie."""
        movie_idx = self.find_movie_index(movie_title)
        
        if movie_idx is None:
            return f"Movie '{movie_title}' not found."
        
        movie = self.cleaned_df.iloc[movie_idx]
        return {
            'title': movie['title'],
            'rating': movie['vote_average'],
            'popularity': movie['popularity'],
            'features_used': movie['tags'][:200] + "..." if len(movie['tags']) > 200 else movie['tags']
        }

# Example usage
def main():
    # Initialize the recommendation system
    # Note: Update these paths to match your actual file locations
    movies_path = "tmdb_5000_movies.csv"  # Update this path
    credits_path = "tmdb_5000_credits.csv"  # Update this path
    
    recommender = MovieRecommendationSystem(movies_path, credits_path)
    
    try:
        # Load and preprocess data
        recommender.load_and_preprocess_data()
        
        # Build similarity matrix
        recommender.build_similarity_matrix()
        
        # Test recommendations
        print("\n" + "="*50)
        print("MOVIE RECOMMENDATION SYSTEM")
        print("="*50)
        
        # Example recommendations
        test_movies = ["Avatar", "The Dark Knight", "Inception"]
        
        for movie in test_movies:
            print(f"\n🎬 Recommendations for '{movie}':")
            print("-" * 40)
            
            result = recommender.get_recommendations(movie, num_recommendations=5)
            
            if 'error' in result:
                print(result['error'])
                if 'suggestions' in result:
                    print("Did you mean:", ', '.join(result['suggestions']))
            else:
                print(f"Based on: {result['input_movie']}")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"{i}. {rec['title']} (Rating: {rec['rating']}, "
                          f"Similarity: {rec['similarity_score']})")
    
    except FileNotFoundError:
        print("Error: CSV files not found. Please update the file paths in the main() function.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()