import pandas as pd

def extract_temporal_features(df):
    # Parse 'published_at' column to handle potential variations in datetime formats
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    
    # Extract day of the week while handling missing or invalid dates
    df['day_of_week'] = df['published_at'].dt.day_name()
    
    # Handle missing or invalid dates with NaT (Not a Time)
    df['day_of_week'].fillna('Unknown', inplace=True)
    
    # Additional feature extraction operations can be added here
    
    return df
