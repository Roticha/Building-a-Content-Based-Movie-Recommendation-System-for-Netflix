# Building-a-Content-Based-Movie-Recommendation-System-for-Netflix

Phase: 4
Group: 13

Group Members:
- Sylvia Manono
- Amos Kipngetich
- Angela Maina
- Charles Ndegwa
- Sandra Koech
- Gloria Tisnanga
- Alex Miningwa

Student Pace: Part time

Scheduled Project Review Date/Time: October 14, 2024

Instructor Name: Samuel G. Mwangi

## Summary
### Business and Data Understanding

The stakeholder for this project is Netflix, a global streaming platform with a vast and diverse movie catalog. Netflix's mission is to provide personalized content to its users, ensuring they stay engaged and satisfied with their viewing experience.

With thousands of movies available on Netflix, users often face difficulty in discovering new content that aligns with their preferences, leading to decision fatigue and potentially lower engagement. Netflix wants to enhance its recommendation engine by suggesting movies similar to those users have already enjoyed, based on the content and genre of the films. 

The dataset used includes movie titles, genres, and descriptions, which are ideal for developing a recommendation system based on content similarity.

### Objective

The main objective for this project is to build a content-based recommendation system for Netflix that can suggest relevant films to users, keeping them engaged on the platform and increasing viewing time. This system will help Netflix continue to offer a personalized and enjoyable user experience.


### Data Preparation

Data preparation involved combining three key columns—movie title, genre, and description—into a single feature. This combination allowed us to create a robust representation of each movie’s content. Missing values were handled by removing incomplete entries, ensuring data quality. We applied the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to the combined text to convert the descriptions into numerical features. This method was chosen because it effectively transforms text data into a format that can be used for similarity analysis. Pandas was used for data manipulation, and Scikit-learn was employed for text vectorization and similarity computation.

### Modeling

We employed cosine similarity as the key metric to measure the closeness between movies based on their textual content. Using the cosine similarity score, we built a function that generates the top 5 movie recommendations for any given movie. The Surprise library and Singular Value Decomposition (SVD) were considered but ultimately not used here, as this model relies on content features rather than user ratings.

### Evaluation

The model was tested by retrieving the top 5 recommendations for specific movies. For example, when queried with Casanova (2015), the model suggested other films such as Das Casanova-Projekt (1981) and Peur de rien (2015). The results demonstrate the ability to recommend movies based on shared content characteristics. Although no formal performance metric (e.g., RMSE) was applicable due to the content-based nature of the model, the quality of recommendations was visually inspected for relevance.

### Limitations and Recommendations

The current recommendation system relies purely on movie metadata and descriptions. It does not take into account user preferences or ratings, which could limit its effectiveness for personalized recommendations. Future improvements could involve integrating collaborative filtering techniques with user rating data to develop a hybrid system that combines both content and user interactions for more accurate recommendations.

## 1. Data Understanding
In this section, we will load the data, understand its structure, and check for missing values.

```
import pandas as pd

# Load the dataset
movie_data = pd.read_csv('movie_descriptions.csv')

# Show dataset info
movie_data.info()

# Display basic statistics
print(movie_data.describe())

# Sample of the dataset
movie_data.sample(5)

```

The dataset contains the following columns:

- id: A unique identifier for each movie.
- title: The title of the movie.
- genre: The genre of the movie (e.g., documentary, drama, comedy).
- desc: A description or summary of the movie.

```

# List all unique movie titles
print(movie_data['title'].unique())


```

## 2. Data Preparation

In this section, We will combine the relevant columns into a single feature for content-based filtering. We will also apply the TF-IDF vectorizer to process text data.

```

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine the title, genre, and description into one column
movie_data['combined_features'] = movie_data['title'] + " " + movie_data['genre'] + " " + movie_data['desc']

# Use TF-IDF Vectorizer to transform the combined text data
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf.fit_transform(movie_data['combined_features'])

# Print the shape of the TF-IDF matrix
print(tfidf_matrix.shape)

```

## 3. Modelling

Here we will calculate cosine similarity between movies based on the TF-IDF features and create the recommendation function.

```

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Standardize movie titles in the dataset
movie_data['title'] = movie_data['title'].str.lower().str.strip()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Standardize the input title
    title = title.lower().strip()

    # Find the movie index with a case-insensitive match
    idx = movie_data[movie_data['title'] == title].index
    
    if len(idx) == 0:
        return "Movie not found in the dataset."
    
    idx = idx[0]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 5 most similar movies
    sim_scores = sim_scores[1:6]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the titles of the recommended movies
    return movie_data['title'].iloc[movie_indices]
    
```
## 4. Evaluating the Model

We will test our model on a few movies and inspect the recommendations.

```

# Test with a few movie titles
print(get_recommendations('Casanova (2015)'))
print(get_recommendations('Burning Man (2007)'))

```

Our model works perfectly. A Netflix User searching for movie "Cassanova (2015) will have the following top 5 suggestions:
 - das casanova-projekt (1981)
 - peur de rien (2015)
 - exiled in america (1992)
 - "temptation of an angel" (2009)
 - mes nuits feront écho (2016)


For Netflix User searching for movie "Burning Man (2007)", they will have the following top 5 suggestions:
 - ritualnation (2000)
 - a day in black and white (2001)
 - rock hard (2003/i)
 - black power: america's armed resistance (2016)
 - short cut road (2003)
 
 
 ## 5. Results and Limitations

The content-based recommendation system successfully generated movie suggestions based on cosine similarity of TF-IDF vectors. The results demonstrate that the model can find relevant movies based on shared characteristics like genre and description.

### Limitations:
- The model relies solely on text data (movie descriptions and genres), and does not take into account user preferences or historical ratings.
- Sparsity in descriptions may limit the quality of recommendations for movies with minimal information.
- Future enhancements could involve integrating user data for collaborative filtering or building a hybrid recommendation system.

## 6. Conclusion and Recommendations
This project successfully developed a content-based recommendation system using TF-IDF vectorization and cosine similarity to suggest similar movies. The model can be further enhanced by integrating user interaction data, such as ratings, to create a hybrid model combining collaborative and content-based filtering for more personalized recommendations.
