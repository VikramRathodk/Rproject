import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import clean_data
from feature_engineering import extract_temporal_features
# from algorithms import tfidf_ranking, weighted_popularity_ranking
from algorithms import  weighted_popularity_ranking
from machine_learning import train_lasso_regression_model
from sklearn.metrics.pairwise import cosine_similarity


# Read the dataset
df = pd.read_csv('../data/articles_data.csv')

# Clean the data
df_cleaned = clean_data(df)

# Extract temporal features
df_with_features = extract_temporal_features(df_cleaned)

# Perform popularity-based ranking
ranked_articles = weighted_popularity_ranking(df_with_features)


# Train a machine learning model
trained_model = train_lasso_regression_model(df_with_features,2)
print("_________------------------------------------------------______________")
test_data = {
    'engagement_reaction_count': [10, 150, 200, 50, 80],  # Add more test values here
    'engagement_comment_count': [150, 80, 120, 20, 110],  # Add more test values for each feature
    'engagement_share_count': [130, 40, 50, 70, 90]  # Add more test values for each feature
}
test_df = pd.DataFrame(test_data)

# Use the trained model to make predictions on the test data
trained_model = train_lasso_regression_model(df_with_features,2)  # Train the model on your original data
predictions = trained_model.predict(test_df)  # Use the trained model to predict on test data
test_df['weighted_popularity_score'] = predictions

# Display the predictions
print("Predictions:", predictions)
# Display the articles along with their predicted popularity scores
# Sort articles based on predicted popularity scores to rank them
ranked_test_df = test_df.sort_values(by='weighted_popularity_score', ascending=False)

# Display the ranked articles along with their predicted popularity scores
print("Ranked Test Articles with Predicted Popularity Scores:")
print(ranked_test_df[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count', 'weighted_popularity_score']])

top_articles_weighted = ranked_articles[['title', 'weighted_popularity_score']].head(20)

print("Top Articles by Weighted Popularity:")
print(top_articles_weighted)


# # Perform TF-IDF ranking
# top_articles_tfidf = tfidf_ranking(df)

# print("\nTop Articles by TF-IDF Ranking:")
# print(top_articles_tfidf)


# print("-------------------------------------------------------")

# # Assuming top_articles_weighted and top_articles_tfidf are DataFrames containing top articles by each method
# common_articles = pd.merge(top_articles_weighted, top_articles_tfidf, on='title', how='inner')
# num_common_articles = len(common_articles)
# print("Number of Common Articles:", num_common_articles)

# # Convert titles to sets for both methods
# titles_weighted = set(top_articles_weighted['title'])
# titles_tfidf = set(top_articles_tfidf['title'])

# # Calculate Jaccard similarity
# jaccard_similarity = len(titles_weighted.intersection(titles_tfidf)) / len(titles_weighted.union(titles_tfidf))
# print("Jaccard Similarity:", jaccard_similarity)

# # Extract TF-IDF scores for the top articles obtained by each method
# tfidf_scores_weighted = top_articles_weighted['weighted_popularity_score'].to_numpy().reshape(-1, 1)
# tfidf_scores_tfidf = top_articles_tfidf['tfidf_score'].to_numpy().reshape(-1, 1)

# # Compute cosine similarity between the TF-IDF scores
# cosine_sim = cosine_similarity(tfidf_scores_weighted, tfidf_scores_tfidf)
# print("Cosine Similarity between Top Articles:")
# print(cosine_sim)
# print("-------------------------------------------------------")
# # Plotting weighted popularity scores
# plt.figure(figsize=(20, 6))
# plt.barh(top_articles_weighted['title'], top_articles_weighted['weighted_popularity_score'], color='skyblue', label='Weighted Popularity')
# plt.xlabel('Weighted Popularity Score')
# plt.title('Top 20 Articles by Weighted Popularity')
# plt.gca().invert_yaxis() 
# plt.tight_layout()

# # Plotting TF-IDF scores
# plt.figure(figsize=(20, 6))
# plt.barh(top_articles_tfidf['title'], top_articles_tfidf['tfidf_score'], color='lightgreen', label='TF-IDF Score')
# plt.xlabel('TF-IDF Score')
# plt.title('Top 20 Articles by TF-IDF Ranking')
# plt.gca().invert_yaxis() 
# plt.tight_layout()

# plt.show()


