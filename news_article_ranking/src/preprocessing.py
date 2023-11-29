import pandas as pd

def clean_data(df):
    # Data cleaning operations
    # Example: Dropping duplicates and handling missing values
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['engagement_reaction_count', 'engagement_comment_count', 'engagement_share_count'], inplace=True)
    # Add more cleaning operations as needed
    return df
