# # import streamlit as st
# # import pandas as pd

# # # Load your dataset or create an empty DataFrame if no data is available initially
# # main_data = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# # def calculate_popularity_score(df):
# #     # Calculate the popularity score based on engagement metrics
# #     weights = {
# #         'engagement_reaction_count': 1,
# #         'engagement_comment_count': 2,
# #         'engagement_share_count': 1.5
# #         # Add other engagement metrics and their weights as needed
# #     }

# #     df['popularity_score'] = (
# #         df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
# #         df['engagement_comment_count'] * weights['engagement_comment_count'] +
# #         df['engagement_share_count'] * weights['engagement_share_count']
# #         # Calculate overall popularity score based on engagement metrics
# #     )
    
# #     # Convert 'popularity_score' column to numeric type
# #     df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
# #     return df

# # def display_sorted_articles(df):
# #     # Display all articles sorted by popularity score
# #     sorted_articles = df.sort_values(by='popularity_score', ascending=False)
# #     return sorted_articles[['title', 'popularity_score']]

# # def main():
# #     # Initialize user_data in session_state if not present
# #     if 'user_data' not in st.session_state:
# #         st.session_state['user_data'] = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# #     st.title('News Article Popularity Ranking')

# #     # Input for adding new articles and their engagement metrics
# #     st.sidebar.header('Add New Article')
# #     new_title = st.sidebar.text_input('Enter Article Title')
# #     new_reaction_count = st.sidebar.number_input('Enter Reaction Count', min_value=0)
# #     new_comment_count = st.sidebar.number_input('Enter Comment Count', min_value=0)
# #     new_share_count = st.sidebar.number_input('Enter Share Count', min_value=0)

# #     if st.sidebar.button('Add Article'):
# #         new_article = {
# #             'title': new_title,
# #             'engagement_reaction_count': new_reaction_count,
# #             'engagement_comment_count': new_comment_count,
# #             'engagement_share_count': new_share_count
# #         }
# #         # Add the new article to the user_data in session_state
# #         st.session_state['user_data'] = pd.concat([st.session_state['user_data'], pd.DataFrame([new_article])], ignore_index=True)
# #         st.success('Article added successfully!')

# #     # Combine and calculate popularity scores for all articles
# #     combined_data = pd.concat([main_data, st.session_state['user_data']], ignore_index=True)
# #     combined_data_with_score = calculate_popularity_score(combined_data)

# #     # Display all articles sorted by popularity score
# #     st.write('### All Articles Sorted by Popularity:')
# #     sorted_articles = display_sorted_articles(combined_data_with_score)
# #     st.write(sorted_articles)

# # if __name__ == '__main__':
# #     main()



# import streamlit as st
# import pandas as pd

# # Load main_data from articles_data.csv or create an empty DataFrame if the file doesn't exist
# main_data_path = 'articles_data.csv'
# try:
#     main_data = pd.read_csv(main_data_path)
# except FileNotFoundError:
#     main_data = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# def calculate_popularity_score(df):
#     # Calculate the popularity score based on engagement metrics
#     weights = {
#         'engagement_reaction_count': 1,
#         'engagement_comment_count': 2,
#         'engagement_share_count': 1.5
#         # Add other engagement metrics and their weights as needed
#     }

#     df['popularity_score'] = (
#         df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
#         df['engagement_comment_count'] * weights['engagement_comment_count'] +
#         df['engagement_share_count'] * weights['engagement_share_count']
#         # Calculate overall popularity score based on engagement metrics
#     )
    
#     # Convert 'popularity_score' column to numeric type
#     df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
#     return df

# def display_user_articles_ranking(df):
#     # Calculate popularity scores for user_data and display its ranking
#     df_with_score = calculate_popularity_score(df)
#     sorted_user_articles = df_with_score.sort_values(by='popularity_score', ascending=False)
#     return sorted_user_articles[['title', 'popularity_score', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']]

# def main():
#     # Initialize user_data in session_state if not present
#     if 'user_data' not in st.session_state:
#         st.session_state['user_data'] = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

#     st.title('News Article Popularity Ranking')

#     # Input for adding new articles and their engagement metrics
#     st.sidebar.header('Add New Article')
#     new_title = st.sidebar.text_input('Enter Article Title')
#     new_reaction_count = st.sidebar.number_input('Enter Reaction Count', min_value=0)
#     new_comment_count = st.sidebar.number_input('Enter Comment Count', min_value=0)
#     new_share_count = st.sidebar.number_input('Enter Share Count', min_value=0)

#     if st.sidebar.button('Add Article'):
#         new_article = {
#             'title': new_title,
#             'engagement_reaction_count': new_reaction_count,
#             'engagement_comment_count': new_comment_count,
#             'engagement_share_count': new_share_count
#         }
#         # Add the new article to the user_data in session_state
#         st.session_state['user_data'] = pd.concat([st.session_state['user_data'], pd.DataFrame([new_article])], ignore_index=True)
#         st.success('Article added successfully!')

#     # Display user-added articles ranking based on their popularity
#     if not st.session_state['user_data'].empty:
#         st.write('### User-added Articles Ranking Based on Popularity:')
#         user_articles_ranking = display_user_articles_ranking(st.session_state['user_data'])
#         st.write(user_articles_ranking)
#     else:
#         st.write('Add user articles to see their ranking.')

# if __name__ == '__main__':
#     main()


#Working code

# import streamlit as st
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler

# # Load main_data from articles_data.csv or create an empty DataFrame if the file doesn't exist
# main_data_path = '../data/articles_data.csv'
# try:
#     main_data = pd.read_csv(main_data_path)
# except FileNotFoundError:
#     main_data = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# def calculate_popularity_score(df):
#     # Calculate the popularity score based on engagement metrics
#     weights = {
#         'engagement_reaction_count': 1,
#         'engagement_comment_count': 2,
#         'engagement_share_count': 1.5
#         # Add other engagement metrics and their weights as needed
#     }

#     df['popularity_score'] = (
#         df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
#         df['engagement_comment_count'] * weights['engagement_comment_count'] +
#         df['engagement_share_count'] * weights['engagement_share_count']
#         # Calculate overall popularity score based on engagement metrics
#     )
    
#     # Convert 'popularity_score' column to numeric type
#     df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
#     return df

# def calculate_similarity(main_df, user_df):
#     # Replace NaN values with zeros
#     main_df_filled = main_df.fillna(0)
#     user_df_filled = user_df.fillna(0)

#     # Select only engagement metrics columns for similarity calculation
#     main_engagement = main_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values
#     user_engagement = user_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values

#     # Normalize engagement metrics using Min-Max scaling
#     scaler = MinMaxScaler()
#     main_engagement_scaled = scaler.fit_transform(main_engagement)
#     user_engagement_scaled = scaler.transform(user_engagement)

#     # Compute cosine similarity between user_data and main_data after handling NaN values
#     similarity_scores = cosine_similarity(user_engagement_scaled, main_engagement_scaled)
#     return similarity_scores

# def display_user_articles_ranking(df, similarity_scores):
#     # Calculate popularity scores for user_data
#     df_with_score = calculate_popularity_score(df)

#     # Calculate the average similarity score for each user article
#     avg_similarity_scores = similarity_scores.mean(axis=1)

#     # Combine popularity scores and average similarity scores
#     df_with_score['average_similarity_score'] = avg_similarity_scores

#     # Sort user-added articles based on popularity and similarity scores
#     sorted_user_articles = df_with_score.sort_values(by=['popularity_score', 'average_similarity_score'], ascending=[False, False])
#     return sorted_user_articles[['title', 'popularity_score', 'average_similarity_score', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']]

# def main():
#     # Initialize user_data in session_state if not present
#     if 'user_data' not in st.session_state:
#         st.session_state['user_data'] = pd.DataFrame(columns=['title', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

#     st.title('News Article Popularity Ranking')

#     # Input for adding new articles and their engagement metrics
#     st.sidebar.header('Add New Article')
#     new_title = st.sidebar.text_input('Enter Article Title')
#     new_reaction_count = st.sidebar.number_input('Enter Reaction Count', min_value=0)
#     new_comment_count = st.sidebar.number_input('Enter Comment Count', min_value=0)
#     new_share_count = st.sidebar.number_input('Enter Share Count', min_value=0)

#     if st.sidebar.button('Add Article'):
#         new_article = {
#             'title': new_title,
#             'engagement_reaction_count': new_reaction_count,
#             'engagement_comment_count': new_comment_count,
#             'engagement_share_count': new_share_count
#         }
#         # Add the new article to the user_data in session_state
#         st.session_state['user_data'] = pd.concat([st.session_state['user_data'], pd.DataFrame([new_article])], ignore_index=True)
#         st.success('Article added successfully!')

#     # Display user-added articles ranking based on popularity and similarity to main_data
#     if not st.session_state['user_data'].empty:
#         similarity_scores = calculate_similarity(main_data, st.session_state['user_data'])
#         st.write('### User-added Articles Ranking Based on Popularity and Similarity to Main Data:')
#         user_articles_ranking = display_user_articles_ranking(st.session_state['user_data'], similarity_scores)
#         st.write(user_articles_ranking)
#     else:
#         st.write('Add user articles to see their ranking.')

# if __name__ == '__main__':
#     main()



# import streamlit as st
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load main_data from articles_data.csv or create an empty DataFrame if the file doesn't exist
# main_data_path = '../data/articles_data.csv'
# try:
#     main_data = pd.read_csv(main_data_path)
# except FileNotFoundError:
#     main_data = pd.DataFrame(columns=['title','content', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# def calculate_popularity_score(df):
#     # Calculate the popularity score based on engagement metrics
#     weights = {
#         'engagement_reaction_count': 1,
#         'engagement_comment_count': 2,
#         'engagement_share_count': 1.5
#         # Add other engagement metrics and their weights as needed
#     }

#     df['popularity_score'] = (
#         df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
#         df['engagement_comment_count'] * weights['engagement_comment_count'] +
#         df['engagement_share_count'] * weights['engagement_share_count']
#         # Calculate overall popularity score based on engagement metrics
#     )
    
#     # Convert 'popularity_score' column to numeric type
#     df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
#     return df

# def calculate_similarity(main_df, user_df):
#     # Replace NaN values with zeros
#     main_df_filled = main_df.fillna(0)
#     user_df_filled = user_df.fillna(0)

#     # Select only engagement metrics columns for similarity calculation
#     main_engagement = main_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values
#     user_engagement = user_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values

#     # Normalize engagement metrics using Min-Max scaling
#     scaler = MinMaxScaler()
#     main_engagement_scaled = scaler.fit_transform(main_engagement)
#     user_engagement_scaled = scaler.transform(user_engagement)

#     # Compute cosine similarity between user_data and main_data after handling NaN values
#     similarity_scores = cosine_similarity(user_engagement_scaled, main_engagement_scaled)
#     return similarity_scores

# def display_user_articles_ranking(df, similarity_scores):
#     # Calculate popularity scores for user_data
#     df_with_score = calculate_popularity_score(df)

#     # Calculate the average similarity score for each user article
#     avg_similarity_scores = similarity_scores.mean(axis=1)

#     # Combine popularity scores and average similarity scores
#     df_with_score['average_similarity_score'] = avg_similarity_scores

#     # Sort user-added articles based on popularity and similarity scores
#     sorted_user_articles = df_with_score.sort_values(by=['popularity_score', 'average_similarity_score'], ascending=[False, False])
#     return sorted_user_articles[['title', 'popularity_score', 'average_similarity_score', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']]

# def tfidf_ranking(df):
#     # Combine 'title' and 'content' columns into a single text column for TF-IDF calculation
#     df['title_content'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
#     # TF-IDF vectorization considering both 'title' and 'content' features
#     tfidf = TfidfVectorizer(max_features=5000)
#     tfidf_matrix = tfidf.fit_transform(df['title_content'])

#     # Calculate maximum TF-IDF scores across the text for ranking
#     df['tfidf_score'] = tfidf_matrix.max(axis=1).toarray().flatten()

#     return df[['title_content', 'tfidf_score']].sort_values(by='tfidf_score', ascending=False)
# def main():
#     # Initialize user_data in session_state if not present
#     if 'user_data' not in st.session_state:
#         st.session_state['user_data'] = pd.DataFrame(columns=['title', 'content', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

#     st.title('News Article Popularity Ranking')

#     # Input for adding new articles and their engagement metrics
#     st.sidebar.header('Add New Article')
#     new_title = st.sidebar.text_input('Enter Article Title')
#     new_content = st.sidebar.text_area('Enter Article Content')
#     new_reaction_count = st.sidebar.number_input('Enter Reaction Count', min_value=0)
#     new_comment_count = st.sidebar.number_input('Enter Comment Count', min_value=0)
#     new_share_count = st.sidebar.number_input('Enter Share Count', min_value=0)

#     if st.sidebar.button('Add Article'):
#         new_article = {
#             'title': new_title,
#             'content': new_content,
#             'engagement_reaction_count': new_reaction_count,
#             'engagement_comment_count': new_comment_count,
#             'engagement_share_count': new_share_count
#         }
#         # Add the new article to the user_data in session_state
#         st.session_state['user_data'] = pd.concat([st.session_state['user_data'], pd.DataFrame([new_article])], ignore_index=True)
#         st.success('Article added successfully!')

#     # Display user-added articles ranking based on popularity and similarity to main_data
#     if not st.session_state['user_data'].empty:
#         similarity_scores = calculate_similarity(main_data, st.session_state['user_data'])
#         st.write('### User-added Articles Ranking Based on Popularity and Similarity to Main Data:')
#         user_articles_ranking = display_user_articles_ranking(st.session_state['user_data'], similarity_scores)
#         st.write(user_articles_ranking)

#         st.write('### TF-IDF Ranking Based on Title and Content:')
#         tfidf_articles_ranking = tfidf_ranking(st.session_state['user_data'])
#         st.write(tfidf_articles_ranking)
#     else:
#         st.write('Add user articles to see their ranking.')

# if __name__ == '__main__':
#     main()  



import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load main_data from articles_data.csv or create an empty DataFrame if the file doesn't exist
main_data_path = '../data/articles_data.csv'
try:
    main_data = pd.read_csv(main_data_path)
except FileNotFoundError:
    main_data = pd.DataFrame(columns=['title', 'content', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

# Functions for calculating popularity score, similarity, and TF-IDF ranking

def calculate_popularity_score(df):
    # Calculate the popularity score based on engagement metrics
    weights = {
        'engagement_reaction_count': 1,
        'engagement_comment_count': 2,
        'engagement_share_count': 1.5
        # Add other engagement metrics and their weights as needed
    }

    df['popularity_score'] = (
        df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
        df['engagement_comment_count'] * weights['engagement_comment_count'] +
        df['engagement_share_count'] * weights['engagement_share_count']
        # Calculate overall popularity score based on engagement metrics
    )
    
    # Convert 'popularity_score' column to numeric type
    df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
    return df

def calculate_similarity(main_df, user_df):
    # Replace NaN values with zeros
    main_df_filled = main_df.fillna(0)
    user_df_filled = user_df.fillna(0)

    # Select only engagement metrics columns for similarity calculation
    main_engagement = main_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values
    user_engagement = user_df_filled[['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']].values

    # Normalize engagement metrics using Min-Max scaling
    scaler = MinMaxScaler()
    main_engagement_scaled = scaler.fit_transform(main_engagement)
    user_engagement_scaled = scaler.transform(user_engagement)

    # Compute cosine similarity between user_data and main_data after handling NaN values
    similarity_scores = cosine_similarity(user_engagement_scaled, main_engagement_scaled)
    return similarity_scores

def display_user_articles_ranking(df, similarity_scores):
    # Calculate popularity scores for user_data
    df_with_score = calculate_popularity_score(df)

    # Calculate the average similarity score for each user article
    avg_similarity_scores = similarity_scores.mean(axis=1)

    # Combine popularity scores and average similarity scores
    df_with_score['average_similarity_score'] = avg_similarity_scores

    # Sort user-added articles based on popularity and similarity scores
    sorted_user_articles = df_with_score.sort_values(by=['popularity_score', 'average_similarity_score'], ascending=[False, False])
    return sorted_user_articles[['title', 'popularity_score', 'average_similarity_score', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']]

def tfidf_ranking(df):
    # Combine 'title' and 'content' columns into a single text column for TF-IDF calculation
    df['title_content'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # TF-IDF vectorization considering both 'title' and 'content' features
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['title_content'])

    # Calculate maximum TF-IDF scores across the text for ranking
    df['tfidf_score'] = tfidf_matrix.max(axis=1).toarray().flatten()

    return df[['title_content', 'tfidf_score']].sort_values(by='tfidf_score', ascending=False)

def neural_network_ranking(main_df, user_df):
    weights = {
        'engagement_reaction_count': 1,
        'engagement_comment_count': 2,
        'engagement_share_count': 1.5
    }

    main_df['weighted_popularity_score'] = (
        main_df['engagement_reaction_count'] * weights['engagement_reaction_count'] +
        main_df['engagement_comment_count'] * weights['engagement_comment_count'] +
        main_df['engagement_share_count'] * weights['engagement_share_count']
    )

    features = ['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count']
    target = 'weighted_popularity_score'

    X_train = main_df[features]
    y_train = main_df[target]

    X_test = user_df[features]  # Extract features from user data
    # You can select your target column based on your user data structure
    y_test = user_df['popularity_score']  # Change this to match your actual target column in user_df

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input data to be 3D for LSTM input [samples, timesteps, features]
    X_train_scaled_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_scaled_lstm.shape[1], X_train_scaled_lstm.shape[2]), return_sequences=True))
    model.add(Dropout(0.8))  # Adding dropout for regularization
    model.add(LSTM(32))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.fit(X_train_scaled_lstm, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)  # Adding validation split

    loss, mae = model.evaluate(X_test_scaled_lstm, y_test, verbose=0)
    st.write(f"Mean Absolute Error (MAE) of LSTM model on user data: {mae}")

    user_df_scaled = scaler.transform(X_test)
    neural_network_score = model.predict(user_df_scaled.reshape((user_df_scaled.shape[0], 1, user_df_scaled.shape[1])))

    user_df['neural_network_score_lstm'] = neural_network_score

    user_df_sorted_by_lstm_score = user_df.sort_values(by='neural_network_score_lstm', ascending=False)

    return user_df_sorted_by_lstm_score[['title', 'neural_network_score_lstm']]


def main():
    # Initialize user_data in session_state if not present
    if 'user_data' not in st.session_state:
        st.session_state['user_data'] = pd.DataFrame(columns=['title', 'content', 'engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'])

    st.title('News Article Popularity Ranking')

    # Input for adding new articles and their engagement metrics
    st.sidebar.header('Add New Article')
    new_title = st.sidebar.text_input('Enter Article Title')
    new_content = st.sidebar.text_area('Enter Article Content')
    new_reaction_count = st.sidebar.number_input('Enter Reaction Count', min_value=0)
    new_comment_count = st.sidebar.number_input('Enter Comment Count', min_value=0)
    new_share_count = st.sidebar.number_input('Enter Share Count', min_value=0)

    if st.sidebar.button('Add Article'):
        new_article = {
            'title': new_title,
            'content': new_content,
            'engagement_reaction_count': new_reaction_count,
            'engagement_comment_count': new_comment_count,
            'engagement_share_count': new_share_count
        }
        # Add the new article to the user_data in session_state
        st.session_state['user_data'] = pd.concat([st.session_state['user_data'], pd.DataFrame([new_article])], ignore_index=True)
        st.success('Article added successfully!')

    # Display user-added articles ranking based on popularity and similarity to main_data
    if not st.session_state['user_data'].empty:
        similarity_scores = calculate_similarity(main_data, st.session_state['user_data'])
        st.write('### User-added Articles Ranking Based on Popularity and Similarity to Main Data:')
        user_articles_ranking = display_user_articles_ranking(st.session_state['user_data'], similarity_scores)
        st.write(user_articles_ranking)

        st.write('### TF-IDF Ranking Based on Title and Content:')
        tfidf_articles_ranking = tfidf_ranking(st.session_state['user_data'])
        st.write(tfidf_articles_ranking)

        st.write('### Neural Network (LSTM) Ranking:')
        ranked_user_articles = neural_network_ranking(main_data, st.session_state['user_data'])
        st.write(ranked_user_articles)

    else:
        st.write('Add user articles to see their ranking.')

if __name__ == '__main__':
    main()
