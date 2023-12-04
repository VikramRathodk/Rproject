# import pandas as pd
# import matplotlib.pyplot as plt

# from preprocessing import clean_data
# from feature_engineering import extract_temporal_features
# # from algorithms import tfidf_ranking, weighted_popularity_ranking
# from algorithms import  weighted_popularity_ranking
# from machine_learning import train_lasso_regression_model
# from sklearn.metrics.pairwise import cosine_similarity


# # Read the dataset
# df = pd.read_csv('../data/articles_data.csv')

# # Clean the data
# df_cleaned = clean_data(df)

# # Extract temporal features
# df_with_features = extract_temporal_features(df_cleaned)

# # Perform popularity-based ranking
# ranked_articles = weighted_popularity_ranking(df_with_features)


# # Train a machine learning model
# trained_model = train_lasso_regression_model(df_with_features,2)


# print("_________------------------------------------------------______________")
# test_data = {
#     'engagement_reaction_count': [10, 150, 200, 50, 80],  # Add more test values here
#     'engagement_comment_count': [150, 80, 120, 20, 110],  # Add more test values for each feature
#     'engagement_share_count': [130, 40, 50, 70, 90]  # Add more test values for each feature
# }
# test_df = pd.DataFrame(test_data)

# # Use the trained model to make predictions on the test data
# trained_model = train_lasso_regression_model(df_with_features,2)  # Train the model on your original data
# predictions = trained_model.predict(test_df)  # Use the trained model to predict on test data
# test_df['weighted_popularity_score'] = predictions

# # Display the predictions
# print("Predictions:", predictions)
# # Display the articles along with their predicted popularity scores
# # Sort articles based on predicted popularity scores to rank them
# ranked_test_df = test_df.sort_values(by='weighted_popularity_score', ascending=False)

# # Display the ranked articles along with their predicted popularity scores
# print("Ranked Test Articles with Predicted Popularity Scores:")
# print(ranked_test_df[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count', 'weighted_popularity_score']])

# top_articles_weighted = ranked_articles[['title', 'weighted_popularity_score']].head(20)

# print("Top Articles by Weighted Popularity:")
# print(top_articles_weighted)


# # # Perform TF-IDF ranking
# # top_articles_tfidf = tfidf_ranking(df)

# # print("\nTop Articles by TF-IDF Ranking:")
# # print(top_articles_tfidf)


# # print("-------------------------------------------------------")

# # # Assuming top_articles_weighted and top_articles_tfidf are DataFrames containing top articles by each method
# # common_articles = pd.merge(top_articles_weighted, top_articles_tfidf, on='title', how='inner')
# # num_common_articles = len(common_articles)
# # print("Number of Common Articles:", num_common_articles)

# # # Convert titles to sets for both methods
# # titles_weighted = set(top_articles_weighted['title'])
# # titles_tfidf = set(top_articles_tfidf['title'])

# # # Calculate Jaccard similarity
# # jaccard_similarity = len(titles_weighted.intersection(titles_tfidf)) / len(titles_weighted.union(titles_tfidf))
# # print("Jaccard Similarity:", jaccard_similarity)

# # # Extract TF-IDF scores for the top articles obtained by each method
# # tfidf_scores_weighted = top_articles_weighted['weighted_popularity_score'].to_numpy().reshape(-1, 1)
# # tfidf_scores_tfidf = top_articles_tfidf['tfidf_score'].to_numpy().reshape(-1, 1)

# # # Compute cosine similarity between the TF-IDF scores
# # cosine_sim = cosine_similarity(tfidf_scores_weighted, tfidf_scores_tfidf)
# # print("Cosine Similarity between Top Articles:")
# # print(cosine_sim)
# # print("-------------------------------------------------------")
# # # Plotting weighted popularity scores
# # plt.figure(figsize=(20, 6))
# # plt.barh(top_articles_weighted['title'], top_articles_weighted['weighted_popularity_score'], color='skyblue', label='Weighted Popularity')
# # plt.xlabel('Weighted Popularity Score')
# # plt.title('Top 20 Articles by Weighted Popularity')
# # plt.gca().invert_yaxis() 
# # plt.tight_layout()

# # # Plotting TF-IDF scores
# # plt.figure(figsize=(20, 6))
# # plt.barh(top_articles_tfidf['title'], top_articles_tfidf['tfidf_score'], color='lightgreen', label='TF-IDF Score')
# # plt.xlabel('TF-IDF Score')
# # plt.title('Top 20 Articles by TF-IDF Ranking')
# # plt.gca().invert_yaxis() 
# # plt.tight_layout()

# # plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from algorithms import weighted_popularity_ranking, tfidf_ranking, neural_network_ranking
from machine_learning import train_lasso_regression_model
from feature_engineering import extract_temporal_features
from preprocessing import clean_data

if __name__ == "__main__":
    df = pd.read_csv('../data/articles_data.csv')


    
    ranked_articles_tfidf = tfidf_ranking(df)
    ranked_articles_nn = neural_network_ranking(df)

  
    # Perform rankings using different algorithms
    ranked_articles_weighted = weighted_popularity_ranking(df)
    
    ranked_articles_tfidf = tfidf_ranking(df)
    ranked_articles_nn = neural_network_ranking(df)

    # Extract titles of top-ranked articles for comparison
    top_ranked_titles_weighted = list(ranked_articles_weighted['title'].head(10))
    top_ranked_titles_tfidf = list(ranked_articles_tfidf['title'].head(10))
    top_ranked_titles_nn = list(ranked_articles_nn['title'].head(10))

    # Calculate Kendall's Tau and Spearman's Rank Correlation
    tau_weighted_tfidf, _ = kendalltau(top_ranked_titles_weighted, top_ranked_titles_tfidf)
    rho_weighted_tfidf, _ = spearmanr(top_ranked_titles_weighted, top_ranked_titles_tfidf)

    
    
    tau_tfidf_nn, _ = kendalltau(top_ranked_titles_tfidf, top_ranked_titles_nn)
    rho_tfidf_nn, _ = spearmanr(top_ranked_titles_tfidf, top_ranked_titles_nn)

    print(f"Weighted vs TF-IDF - Kendall's Tau: {tau_weighted_tfidf}, Spearman's Rank Correlation: {rho_weighted_tfidf}")
    print(f"TF-IDF vs Neural Network - Kendall's Tau: {tau_tfidf_nn}, Spearman's Rank Correlation: {rho_tfidf_nn}")

   
    #MAP
# Simulate relevance scores for the top-K items (considering top 10 in this case)
num_top_items = 10
relevant_items = set(top_ranked_titles_weighted[:num_top_items] + 
                     top_ranked_titles_tfidf[:num_top_items] + 
                     top_ranked_titles_nn[:num_top_items])

simulated_relevance_weighted = [(1 if title in relevant_items else 0) for title in ranked_articles_weighted['title']]
simulated_relevance_tfidf = [(1 if title in relevant_items else 0) for title in ranked_articles_tfidf['title']]
simulated_relevance_nn = [(1 if title in relevant_items else 0) for title in ranked_articles_nn['title']]

# Function to calculate Average Precision (AP)
def average_precision(simulated_relevance):
    num_relevant = sum(simulated_relevance)
    if num_relevant == 0:
        return 0

    precision_at_k = []
    num_correct = 0
    for i, rel in enumerate(simulated_relevance, start=1):
        if rel == 1:
            num_correct += 1
            precision_at_k.append(num_correct / i)
    
    return sum(precision_at_k) / num_relevant

# Calculate Average Precision for each ranking
ap_weighted = average_precision(simulated_relevance_weighted)
ap_tfidf = average_precision(simulated_relevance_tfidf)
ap_nn = average_precision(simulated_relevance_nn)

# Calculate Mean Average Precision (MAP)
map_value = (ap_weighted + ap_tfidf + ap_nn) / 3

print(f"MAP - Weighted: {ap_weighted}")
print(f"MAP - TF-IDF: {ap_tfidf}")
print(f"MAP - Neural Network: {ap_nn}")
print(f"Mean Average Precision (MAP) across rankings: {map_value}")

#graph s
# Create a bar plot for Kendall's Tau and Spearman's Rank Correlation
labels = ['Weighted vs TF-IDF', 'TF-IDF vs Neural Network']
kendall_tau_values = [tau_weighted_tfidf, tau_tfidf_nn]
spearman_rho_values = [rho_weighted_tfidf, rho_tfidf_nn]

x = list(range(len(labels)))  # Convert range to list
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar([pos - width/4 for pos in x], kendall_tau_values, width/2, label="Kendall's Tau")
rects2 = ax.bar([pos + width/4 for pos in x], spearman_rho_values, width/2, label="Spearman's Rank Correlation")

ax.set_ylabel('Correlation Coefficient')
ax.set_title('Comparison of Correlation Coefficients')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


#MAP
# Plotting Mean Average Precision (MAP) across rankings
map_values = [ap_weighted, ap_tfidf, ap_nn, map_value]
labels = ['Weighted', 'TF-IDF', 'Neural Network', 'Mean Across Rankings']

plt.bar(labels, map_values)
plt.xlabel('Ranking Algorithms')
plt.ylabel('Mean Average Precision (MAP)')
plt.title('Comparison of MAP across Rankings')
plt.show()
# Print the top-ranked articles for each method
print("Top-ranked articles for Weighted method:")
print(ranked_articles_weighted.head(10))

print("\nTop-ranked articles for TF-IDF method:")
print(ranked_articles_tfidf.head(10))

print("\nTop-ranked articles for Neural Network method:")
print(ranked_articles_nn.head(10))
