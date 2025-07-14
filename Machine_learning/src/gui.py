import streamlit as st
import pandas as pd
import joblib
import os
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import pickle
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')
sia = SentimentIntensityAnalyzer()

text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview',
             'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about']

def apply_tfidf_reg(df, col, max_features=20):
    MODEL_DIR = "sayed"
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory path
    print(f"Applying TF-IDF on: {col}")

    # Load the saved TF-IDF vectorizer
    tfidf = joblib.load(os.path.join(PROJECT_DIR, f"{col}_tfidf_vectorizer_reg.joblib"))

    # Fill NaNs with empty strings and transform using the loaded vectorizer
    tfidf_matrix = tfidf.transform(df[col].fillna(""))

    # Load the column names
    tfidf_columns = joblib.load(os.path.join(PROJECT_DIR, f"{col}_tfidf_columns_reg.joblib"))

    # Create a DataFrame from TF-IDF matrix with the loaded column names
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns)

    return tfidf_df
def prepare_features(df):
    import pandas as pd
    import numpy as np
    import pickle, joblib, os, re, string
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.cluster import KMeans

    target_col = 'review_scores_rating'
    price_columns = ['nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee', 'extra_people']
    for col in price_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    if 'host_response_rate' in df.columns and df['host_response_rate'].dtype == 'object':
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100

    drop_cols = ['host_acceptance_rate', 'square_feet']
    df = df.drop(drop_cols, axis=1)
    df.drop(['host_listings_count'], axis=1, inplace=True)
    df.drop("thumbnail_url", axis = 1, inplace = True)

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Duplicates dropped. New shape: {df.shape}")

    with open(r'D:\fcis\machine\project\modesToReplaceNulls_reg.pkl', 'rb') as file:
        modesForPickle = pickle.load(file)

    print(type(modesForPickle))
    if isinstance(modesForPickle, dict):
        print(modesForPickle.keys())
    else:
        print("The loaded object is not a dictionary.")

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
            df[col] = df[col].fillna(modesForPickle[col])

    df.drop("zipcode", axis = 1, inplace = True)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    x = pd.DataFrame(num_cols, columns=['Numeric Columns'])

    with open(r'D:\fcis\machine\project\pickleFiles\reg_scalerModel.pkl', 'rb') as file:
        scaler= pickle.load(file)

    scaled_data = scaler.transform(df[num_cols])

    with open(r'D:\fcis\machine\project\pickleFiles\re_imputerModel.pkl', 'rb') as file:
        imputer=pickle.load(file)

    imputed_data = imputer.transform(scaled_data)
    df[num_cols] = pd.DataFrame(scaler.inverse_transform(imputed_data), columns=num_cols, index=df.index)

    skewed_cols = ['nightly_price', 'price_per_stay', 'number_of_reviews', 'host_total_listings_count', 'number_of_stays']
    for col in skewed_cols:
        df[f'{col}'] = np.log1p(df[col])

    with open(r'D:\fcis\machine\project\pkls2\lowerUpperOutliersForPickle.pkl', 'rb') as file:
        lowerUpperOutliers= pickle.load(file)

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = np.where(df[col] < lowerUpperOutliers[col][0], lowerUpperOutliers[col][0], df[col])
        df[col] = np.where(df[col] > lowerUpperOutliers[col][1], lowerUpperOutliers[col][1], df[col])

    host_total_listings_count = df.groupby('host_id')['host_id'].count().to_dict()
    df['host_experience'] = np.log1p(df['host_id'].map(lambda x: host_total_listings_count.get(x, 0)))

    host_avg_price = df.groupby('host_id')['price_per_stay'].mean().to_dict()
    df['host_avg_price'] = df['host_id'].map(host_avg_price)

    host_total_reviews = df.groupby('host_id')['number_of_reviews'].sum().to_dict()
    df['host_total_reviews'] = df['host_id'].map(host_total_reviews)

    host_avg_reviews = df.groupby('host_id')['number_of_reviews'].mean()
    df['host_avg_reviews'] = df['host_id'].map(host_avg_reviews)

    host_avg_price = df.groupby('host_id')['nightly_price'].mean()
    df['host_avg_price'] = df['host_id'].map(host_avg_price)
    df['host_total_listings_count_log'] = np.log1p(df['host_total_listings_count'])

    df['host_experience'] = df['host_total_listings_count_log']
    df['price_per_stay'] = np.log1p(df['price_per_stay'])

    df['super_experience'] = df['host_experience'] * (df['host_is_superhost'] == 't').astype(int)
    df['reviews_per_stay'] = df['number_of_reviews'] / (df['number_of_stays'] + 1)
    df['price_per_person'] = df['price_per_stay'] / df['accommodates']
    df['price_diff'] = df['price_per_stay'] - df['nightly_price']

    host_avg_response = df.groupby('host_id')['host_response_rate'].mean()
    df['host_avg_response_rate'] = df['host_id'].map(host_avg_response)

    df['price_per_guest'] = df['price_per_stay'] / df['accommodates']

    with open(r'D:\fcis\machine\project\pkls2\LabelEncoder_for_host_is_superhost.pkl', 'rb') as file:
        le= pickle.load(file)

    df['host_is_superhost'] = le.transform(df['host_is_superhost'])

    df['superhost_reviews_interaction'] = df['host_is_superhost'] * df['number_of_reviews']
    df['superhost_price_interaction'] = df['host_is_superhost'] * df['price_per_stay']
    df['superhost_experience_interaction'] = df['host_is_superhost'] * df['host_experience']
    df['superhost_reviews_interaction2'] = df['host_is_superhost'] * df['host_total_reviews']
    df['reviews_per_listing'] = df['host_total_reviews'] / (df['host_total_listings_count'] + 1)

    cluster_features = df[['host_total_listings_count', 'host_experience', 'host_total_reviews']].fillna(0)
    with open(r'D:\fcis\machine\project\pickleFiles\clusterScaled.pkl', 'rb') as file:
        scaler=pickle.load(file)
    cluster_scaled = scaler.fit_transform(cluster_features)

    with open(r'D:\fcis\machine\project\pickleFiles\kmeanForHostCluster.pkl', 'rb') as file:
        kmeans=pickle.load(file)
    df['host_cluster'] = kmeans.fit_predict(cluster_scaled)

    url_cols = ['listing_url', 'host_url']
    id_cols = ['id', 'host_id']
    cols_to_drop = id_cols + url_cols
    df.drop(cols_to_drop, axis = 1 , inplace = True)

    date_cols = ['host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    reference_date = pd.Timestamp('2023-01-01')
    for col in date_cols:
        if col in df.columns:
            df[f'{col}_days'] = (reference_date - df[col]).dt.days
            if not df[col].isna().all():
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_year'] = df[col].dt.year
            df = df.drop(col, axis=1)

    text_cols = ['name','access' , 'description', 'neighborhood_overview',  'interaction', 'house_rules', 'host_about']
    sia = SentimentIntensityAnalyzer()

    def process_text(text):
        if pd.isna(text):
            return {
                'word_count': 0,
                'sentiment_compound': 0,
                'sentiment_pos': 0,
                'sentiment_neg': 0,
                'sentiment_neu': 0
            }
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        word_count = len(text.split())
        sentiment = sia.polarity_scores(text)
        return {
            'word_count': word_count,
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_neu': sentiment['neu']
        }

    def process_column(df, col):
        print(f"  Processing {col}...")
        text_features = df[col].apply(process_text)
        text_features_df = pd.DataFrame(text_features.tolist())
        text_features_df.columns = [f"{col}_{feat}" for feat in text_features_df.columns]
        return text_features_df

    for col in text_cols:
        if col in df.columns:
            text_features_df = process_column(df, col)
            df = pd.concat([df, text_features_df], axis=1)
            df = df.drop(col, axis=1)

    df['host_sentiment_label'] = df['host_about_sentiment_compound'].apply(
        lambda x: 'positive' if x > 0.5 else 'neutral' if x > 0 else 'negative'
    )
    df['overall_sentiment_score'] = df[['host_about_sentiment_compound']].mean(axis=1)

    cols_to_drop = [col for col in df.columns if col.endswith('_sentiment_neg') or col.endswith('_sentiment_neu') or col.startswith('name_sentiment')]
    df.drop(columns=cols_to_drop, inplace=True)

    if 'number_of_reviews' in df.columns and 'host_is_superhost' in df.columns:
        if df['host_is_superhost'].dtype == 'object':
            df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
        df['superhost_review_interaction'] = df['host_is_superhost'] * df['number_of_reviews']

    df['price_per_person'] = np.log1p(df['price_per_person'])
    df = df.drop(['amenities'], axis=1)
    df.drop(['property_type'], axis=1, inplace=True)
    df.drop(['host_has_profile_pic', 'require_guest_profile_picture','require_guest_phone_verification','requires_license','is_business_travel_ready' ], axis=1, inplace=True)

    binary_cols = ['host_is_superhost',  'host_identity_verified', 'is_location_exact', 'instant_bookable']
    for col in binary_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].map({'t': 1, 'f': 0})

    df = df.drop('city', axis=1)

    if all(col in df.columns for col in ['price_per_stay', 'nightly_price', 'cleaning_fee']):
        df['price_to_nightly_ratio'] = df['price_per_stay'] / (df['nightly_price'] + 0.001)

    if all(col in df.columns for col in ['number_of_reviews', 'number_of_stays']):
        df['review_rate'] = df['number_of_reviews'] / (df['number_of_stays'] + 0.001)

    high_cardinality_cols = ['host_location', 'host_neighbourhood', 'country', 'city', 'street',
                            'neighbourhood', 'neighbourhood_cleansed', 'zipcode', 'market', 'smart_location']
    for col in high_cardinality_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    with open(r'D:\fcis\machine\project\pickleFiles\host_name_target_means.pkl', 'rb') as file:
        target_means= pickle.load(file)
    df['host_name_target_encoded'] = df['host_name'].map(target_means)
    with open(r'D:\fcis\machine\project\pkls2\host_name_value_counts.pkl', 'rb') as file:
        value_counts= pickle.load(file)
    df['host_name_freq_encoded'] = df['host_name'].map(value_counts)
    with open(r'D:\fcis\machine\project\pickleFiles\selected_features.pkl', 'rb') as file:
        selected_features= pickle.load(file)

    df = df[selected_features.tolist() + [target_col]]

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    with open(r'D:\fcis\machine\project\pickleFiles\scaleFeatures.pkl', 'rb') as file:
        scaler= pickle.load(file)

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
    X_scaled_df=X_scaled_df.fillna(0)
    return X_scaled_df, y
  
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Duplicates dropped. New shape: {df.shape}")
    df.drop(['host_listings_count'], axis=1, inplace=True)
    price_columns = ['nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee', 'extra_people']
    for col in price_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

# Convert percentage string to float
    if 'host_response_rate' in df.columns and df['host_response_rate'].dtype == 'object':
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100 
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    with open(r'D:\fcis\machine\project\reg_scalerModel.pkl', 'rb') as f:
        scaler_m = joblib.load(f)

    
    with open(r'D:\fcis\machine\project\re_imputerModel.pkl', 'rb') as f:
        imputer = joblib.load(f)
    scaled_data = scaler_m.transform(df[num_cols])

    imputed_data = imputer.transform(scaled_data)

    df[num_cols] = pd.DataFrame(scaler_m.inverse_transform(imputed_data), columns=num_cols, index=df.index)
    skewed_cols = ['nightly_price', 'price_per_stay', 'number_of_reviews', 'host_total_listings_count', 'number_of_stays']
    for col in skewed_cols:
        df[f'{col}'] = np.log1p(df[col])
    with open(r'D:\fcis\machine\project\modesToReplaceNulls.pkl', 'rb') as f:
        modes = pickle.load(f)
    
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(modes[col])
    # Handle outliers
    numeric_cols = df.select_dtypes(include="number").columns
    with open(r'D:\fcis\machine\project\lowerUpperOutliersForPickle.pkl', 'rb') as f:    
        threshold = joblib.load(f)

    for col in numeric_cols:   
        df[col] = np.where(df[col] < threshold[col][0], threshold[col][0], df[col])
        df[col] = np.where(df[col] > threshold[col][1], threshold[col][1], df[col])
        host_total_listings_count = df.groupby('host_id')['host_id'].count().to_dict()
    df['host_experience'] = np.log1p(df['host_id'].map(lambda x: host_total_listings_count.get(x, 0)))

    host_avg_price = df.groupby('host_id')['price_per_stay'].mean().to_dict()
    df['host_avg_price'] = df['host_id'].map(host_avg_price)

    host_total_reviews = df.groupby('host_id')['number_of_reviews'].sum().to_dict()
    df['host_total_reviews'] = df['host_id'].map(host_total_reviews)

    host_avg_reviews = df.groupby('host_id')['number_of_reviews'].mean()
    df['host_avg_reviews'] = df['host_id'].map(host_avg_reviews)

    host_avg_price = df.groupby('host_id')['nightly_price'].mean()
    df['host_avg_price'] = df['host_id'].map(host_avg_price)
    df['host_total_listings_count_log'] = np.log1p(df['host_total_listings_count'])
    df['host_experience'] = df['host_total_listings_count_log']
    df['super_experience'] = df['host_experience'] * (df['host_is_superhost'] == 't').astype(int)
    df['reviews_per_stay'] = df['number_of_reviews'] / (df['number_of_stays'] + 1)
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    df['superhost_reviews_interaction'] = df['host_is_superhost'] * df['number_of_reviews']
    df['superhost_price_interaction'] = df['host_is_superhost'] * df['price_per_stay']
    df['superhost_experience_interaction'] = df['host_is_superhost'] * df['host_experience']
    df['superhost_reviews_interaction2'] = df['host_is_superhost'] * df['host_total_reviews']
    df['reviews_per_listing'] = df['host_total_reviews'] / (df['host_total_listings_count'] + 1)
    df['price_per_guest'] = df['price_per_stay'] / df['accommodates']

    cluster_features = df[['host_total_listings_count', 'host_experience', 'host_total_reviews']].fillna(0)
    df['price_to_nightly_ratio'] = df['price_per_stay'] / (df['nightly_price'] + 0.001)

    if all(col in df.columns for col in ['number_of_reviews', 'number_of_stays']):
        df['review_rate'] = df['number_of_reviews'] / (df['number_of_stays'] + 0.001)

    with open(r'D:\fcis\machine\project\clusterScaled.pkl', 'rb') as file:
        cluster_scaled = joblib.load(file)
        
    with open(r'D:\fcis\machine\project\kmeanForHostCluster.pkl', 'rb') as file:
        kmeans = joblib.load(file)

    cluster_scaled = cluster_scaled.transform(cluster_features)
    df['host_cluster'] = kmeans.predict(cluster_scaled)

    # Drop URL and ID columns
    url_cols = ['listing_url', 'host_url']
    id_cols = ['id', 'host_id']
    cols_to_drop = id_cols + url_cols
    df.drop(cols_to_drop, axis=1, inplace=True)

    # Date processing
    date_cols = ['host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create features from dates
    reference_date = pd.Timestamp('2023-01-01')
    for col in date_cols:
        if col in df.columns:
            df[f'{col}_days'] = (reference_date - df[col]).dt.days

            if not df[col].isna().all():
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_year'] = df[col].dt.year

            df = df.drop(col, axis=1)
    for col in text_cols:
        if col in df.columns:
            text_features_df = process_column(df, col)

            df = pd.concat([df, text_features_df], axis=1)

            tfidf_df = apply_tfidf_reg(df, col)

            df = pd.concat([df, tfidf_df], axis=1)

            df = df.drop(col, axis=1)
    df['host_sentiment_label'] = df['host_about_sentiment_compound'].apply(
        lambda x: 'positive' if x > 0.5 else 'neutral' if x > 0 else 'negative'
    )

    df['overall_sentiment_score'] = (
        df[['host_about_sentiment_compound']].mean(axis=1)
    )

    cols_to_drop = [
        col for col in df.columns
        if col.endswith('_sentiment_neg') or
        col.endswith('_sentiment_neu') or
        col.startswith('name_sentiment')
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    cols_to_drop = ['space_sentiment_compound', 'house_rules_sentiment_compound']
    df = df.drop(columns=cols_to_drop)
    df['superhost_review_interaction'] = df['host_is_superhost'] * df['number_of_reviews']

    col = 'host_name'           
    target_col = 'review_scores_rating'
    target_means = df.groupby(col)[target_col].mean()
    df[f'{col}_target_encoded'] = df[col].map(target_means)
    value_counts = df[col].value_counts(normalize=True)
    df[f'{col}_freq_encoded'] = df[col].map(value_counts)
    with open(r'D:\fcis\machine\project\LabelEncoder_for_object_columns_reg.pkl', 'rb') as file:
        le= pickle.load(file)
    df['cancellation_policy'] = le.transform(df['cancellation_policy'])
    # Load selected features
    with open(r'D:\fcis\machine\project\selected_features_reg.pkl', 'rb') as f:
        selected_features = joblib.load(f)

    # Prepare final data
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)    

    X = df[selected_features]
    y = df[target_col]
    # Scale features
    with open(r'D:\fcis\machine\project\scaleFeatures_reg.pkl', 'rb') as file:
        scaler = joblib.load(file)
    st.write(X.columns)
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)
    return X_scaled, y        
       
def preprocess_data(df):
    target_col = 'guest_satisfaction'
    price_columns = ['nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee', 'extra_people']
    cols_to_remove = [
        'host_acceptance_rate',
        'host_listings_count',
        'square_feet',
        'thumbnail_url',
        'zipcode'
        ]
    df = df.drop([col for col in cols_to_remove if col in df.columns], axis=1)
    for col in price_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100  
    with open(r'D:\fcis\machine\project\scaler_model.pkl', 'rb') as file:      
        scaler = joblib.load(file)
    with open(r'D:\fcis\machine\project\imputer_model.pkl', 'rb') as file:
        imputer = joblib.load(file)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaled_data = scaler.transform(df[num_cols])
    imputed_data = imputer.transform(scaled_data)
    df[num_cols] = pd.DataFrame(scaler.inverse_transform(imputed_data), columns=num_cols, index=df.index)
    with open(r'D:\fcis\machine\project\dict_missing_values.pkl', 'rb') as file:
        dict_missing_values = pickle.load(file)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col in df.columns:
         df[col] = df[col].fillna(dict_missing_values.get(col, 'missing'))
    with open(r'D:\fcis\machine\project\outlier_thresholds.pkl', 'rb') as file: 
        outlier_thresholds = joblib.load(file)
    for col in num_cols:
        if col in df.columns:
            lower_bound = outlier_thresholds.get(col, {}).get('lower_bound', np.min(df[col]))
            upper_bound = outlier_thresholds.get(col, {}).get('upper_bound', np.max(df[col]))
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    host_total_reviews = df.groupby('host_id')['number_of_reviews'].sum().to_dict()
    df['host_total_reviews'] = df['host_id'].map(host_total_reviews)
    host_total_listings_count = df.groupby('host_id')['host_id'].count().to_dict()
    df['host_experience'] = np.log1p(df['host_id'].map(lambda x: host_total_listings_count.get(x, 0)))
    df['host_total_listings_count_log'] = np.log1p(df['host_total_listings_count'])
    df['host_experience'] = df['host_total_listings_count_log']
    df['price_per_stay'] = np.log1p(df['price_per_stay'])
    df['superhost_reviews_interaction'] = df['host_is_superhost'] * df['number_of_reviews']
    df['superhost_price_interaction'] = df['host_is_superhost'] * df['price_per_stay']
    df['superhost_experience_interaction'] = df['host_is_superhost'] * df['host_experience']
    df['superhost_reviews_interaction2'] = df['host_is_superhost'] * df['host_total_reviews']

# Compute amenity_score if amenities column exists
    cluster_features = df[['host_total_listings_count', 'host_experience', 'host_total_reviews']]
    with open(r'D:\fcis\machine\project\scaler_cluster.pkl', 'rb') as file:
        scaler_cluster = pickle.load(file)
    cluster_scaled = scaler_cluster.transform(cluster_features)        
    with open(r'D:\fcis\machine\project\kmeans_model.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)    
    df['host_cluster'] = kmeans_model.predict(cluster_scaled)
    reference_date = pd.Timestamp('2025-01-01')  # Use a fixed reference date for consistency
    date_cols = ['host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_days'] = (reference_date - df[col]).dt.days
            if not df[col].isna().all():
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_year'] = df[col].dt.year
            df = df.drop(col, axis=1)
    return df
text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview',
             'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about']
sia = SentimentIntensityAnalyzer()
target_col = 'guest_satisfaction'
def process_text(text):
    if pd.isna(text):
        return {
            'word_count': 0,
            'sentiment_compound': 0,
            'sentiment_pos': 0,
            'sentiment_neg': 0,
            'sentiment_neu': 0
        }
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    word_count = len(text.split())
    sentiment = sia.polarity_scores(text)
    return {
        'word_count': word_count,
        'sentiment_compound': sentiment['compound'],
        'sentiment_pos': sentiment['pos'],
        'sentiment_neg': sentiment['neg'],
        'sentiment_neu': sentiment['neu']
    }
def process_column(df, col):
    print(f"Processing {col}...")
    text_features = df[col].apply(process_text)
    text_features_df = pd.DataFrame(text_features.tolist())
    text_features_df.columns = [f"{col}_{feat}" for feat in text_features_df.columns]
    return text_features_df
def apply_tfidf(df, col, max_features=20):
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # يحصل على مسار المجلد الحالي
    try:
        tfidf = joblib.load(os.path.join(PROJECT_DIR, f"{col}_tfidf_vectorizer.joblib"))
        tfidf_matrix = tfidf.transform(df[col].fillna(""))
        tfidf_columns = joblib.load(os.path.join(PROJECT_DIR, f"{col}_tfidf_columns.joblib"))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns)
        return tfidf_df
    except FileNotFoundError:
        st.error(f"TF-IDF model for {col} not found.")
        return pd.DataFrame()
def process_text_columns(df):
    for col in text_cols:
        if col in df.columns:
            text_features_df = process_column(df, col)
            df = pd.concat([df, text_features_df], axis=1)
            tfidf_df = apply_tfidf(df, col)
            df = pd.concat([df, tfidf_df], axis=1)
            df = df.drop(col, axis=1)
    df['overall_sentiment_score'] = (
    df[['summary_sentiment_compound', 'space_sentiment_compound', 'host_about_sentiment_compound']].mean(axis=1)
        )
    df['price_per_person'] = df['price_per_stay'] / df['accommodates']
    manual_mapping = {'Average': 0, 'High': 1, 'Very High': 2}
    if target_col in df.columns:
        df[target_col] = df[target_col].map(manual_mapping)
# Load and apply host_name target encoding
    if 'host_name' in df.columns:
        target_means = df.groupby('host_name')[target_col].mean()
        df['host_name_target_encoded'] = df['host_name'].map(target_means).fillna(0)        
    return df
def feature_selection(df): 
    # Load the model
    with open(r'D:\fcis\machine\project\selected_features.pkl', 'rb') as file:
        selected_features = joblib.load(file)
    # Select features
    X = df[selected_features]
    y = df['guest_satisfaction']
    with open(r'D:\fcis\machine\project\scaler_model2.pkl', 'rb') as file:
        scaler = joblib.load(file)
    X = scaler.transform(X)
    return X,y
models = {
    'logistic': pickle.load(open(r'D:\fcis\machine\project\logistic.pkl', 'rb')),
    'knn': pickle.load(open(r'D:\fcis\machine\project\knn1.pkl', 'rb')),
    'decision': pickle.load(open(r'D:\fcis\machine\project\decision.pkl', 'rb')),
    'svm': pickle.load(open(r'D:\fcis\machine\project\svm.pkl', 'rb')),
    'randomForest': pickle.load(open(r'D:\fcis\machine\project\randomForest.pkl', 'rb'))
}
models_regression = {
    'linearRegression': pickle.load(open(r'D:\fcis\machine\project\pickleFiles\linearRegression.pkl', 'rb')),
    'huberrRegressor': pickle.load(open(r'D:\fcis\machine\project\pickleFiles\huberModel.pkl', 'rb')),
    'DecisionTreeRegressor': pickle.load(open(r'D:\fcis\machine\project\pickleFiles\DecisionTreeRegressor.pkl', 'rb')),
    'ElasticNet': pickle.load(open(r'D:\fcis\machine\project\pickleFiles\elastic_net_model.pkl', 'rb'))
    
    }
with open(r'D:\fcis\machine\project\pickleFiles\X_polyTransform.pkl', 'rb') as file:
    poly= pickle.load(file)
def evaluate_models_regression(models, X, y):
    results = []
    for model_name, model in models.items():
        X=X[model.feature_names_in_]
        y_pred = model.predict(X)
        #calculate r2_score
        r2 = model.score(X, y)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        results.append((model_name, r2, rmse))
    return results    
def evaluate_models(models, X, y):
    results = []
    for model_name, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        results.append((model_name, accuracy))
    return results
# -------------------- Configuration --------------------
st.set_page_config(
    page_title="🏡 Airbnb Satisfaction Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Helper Functions --------------------
def load_lottie_file(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Couldn't load Lottie file: {e}")
        return None

def send_email(name, email, message):
    try:
        # Configure email settings
        sender = "abdoph71@gmail.com"
        password = "sikh pfjp fxpo unge"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = email
        msg['Subject'] = "Thanks for your feedback!"
        
        body = f"""
        Dear {name},
        
        Thank you for contacting us! We've received your message:
        
        {message}
        
        We'll get back to you soon.
        
        Best regards,
        Airbnb Satisfaction Team
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, email, msg.as_string())
            
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False
def send_email_to_me(name, email, message):
    try:
        # Configure email settings
        sender = "abdoph71@gmail.com"
        password = "sikh pfjp fxpo unge"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = email
        msg['Subject'] = "new message from contact form"
        
        body = f"""
        from {name},
        
        new message from contact form:
        {message}
        
        
        Airbnb Satisfaction Team
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, email, msg.as_string())
            
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False    

# -------------------- Data Processing --------------------

# -------------------- UI Components --------------------
def home_page():
    st.title("🏡 Airbnb Guest Satisfaction Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"D:\fcis\machine\project\w.jpg", use_column_width=True)
    with col2:
        lottie_anim = load_lottie_file(r"D:\fcis\machine\project\ini.json")
        if lottie_anim:
            st_lottie(lottie_anim, height=300)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("📤 Upload your Airbnb dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Successfully loaded {len(df)} records")
        
        # Analysis type selection
        analysis_type = st.radio(
            "🔍 Select analysis type:",
            ["Classification", "Regression"],
            horizontal=True
        )
        
        if st.button("🚀 Run Analysis"):
            with st.spinner("🔧 Processing your data..."):
                try:
                    # Data preprocessing
                    

                    # Model evaluation
                    if analysis_type == "Classification":
                        df = preprocess_data(df)
                        df = process_text_columns(df)
                        st.success("✅ Data preprocessing completed!")
                        X , y = feature_selection(df)
                        results = evaluate_models(models, X, y)
                        
                        # Display results
                        st.subheader("📊 Model Performance")
                        for model_name, accuracy in results:
                            st.metric(model_name, f"{accuracy:.2%}")
                        
                        best_model, best_acc = max(results, key=lambda x: x[1])
                        st.success(f"🏆 Best Model: {best_model} ({best_acc:.2%} accuracy)")
                    if analysis_type == "Regression":
                        X,y=prepare_features(df)  
                        st.write(X.isnull().sum()) 
                        st.success("✅ Data preprocessing completed!")
                        results = evaluate_models_regression(models_regression, X, y)
                        st.subheader("📊 Model Performance")
                        for model_name, r2, rmse in results:
                            st.metric(model_name, f"R²: {r2:.2f}, RMSE: {rmse:.2f}")
                        best_model, best_r2, best_rmse = max(results, key=lambda x: x[1])
                        st.success(f"🏆 Best Model: {best_model} (R²: {best_r2:.2f}, RMSE: {best_rmse:.2f})")
                    # Add regression case if needed
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

def graphs_page():
    st.title("📈 Data Visualizations")
    
    tabs = st.tabs(["Category Visualization", "Log Transformation", "Target of Both MS"])
    
    with tabs[0]:
        st.image(r"D:\fcis\machine\project\visualization\category visualization\1.png", caption="Percentage of Missing Values")
        st.image(r"D:\fcis\machine\project\visualization\category visualization\29.png", caption="Location clustering")
        st.image(r"D:\fcis\machine\project\visualization\category visualization\34.png", caption="Sentiment Analysis for house rules")
        st.image(r"D:\fcis\machine\project\visualization\category visualization\35.png", caption="Distribution of amenities")
        st.image(r"D:\fcis\machine\project\visualization\category visualization\36.png", caption="Top 15 amenities")
        st.image(r"D:\fcis\machine\project\visualization\category visualization\37.png", caption="DIstribution of Binary Features")

    with tabs[1]:
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\27.png", caption="Log Transformation of Price_per_Stay")
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\28.png", caption="Log Transformation of Number of Reviews")
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\40.png", caption="Log Transformation of Review Scores Rating")
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\activity_period_distribution.png", caption="activity_period_distribution")
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\activity_score_distribution.png", caption="activity_score_distribution")
        st.image(r"D:\fcis\machine\project\visualization\Log Transformation\host_reviews_relationship.png", caption="host_reviews_relationship")    
    with tabs[2]:
        st.image(r"D:\fcis\machine\project\visualization\target of both MS\confusion1.png", caption="Confusion Matrix")
        st.image(r"D:\fcis\machine\project\visualization\target of both MS\confusion2.png", caption="Confusion Matrix")
        st.image(r"D:\fcis\machine\project\visualization\target of both MS\guest.png", caption="Guest Satisfaction")
        st.image(r"D:\fcis\machine\project\visualization\target of both MS\review score.png", caption="Review Score Rating Boxplot")  

def about_page():
    st.title("ℹ️ About Our Project")
    
    st.markdown("""
    <div style='background-color:#f8f9fa;padding:20px;border-radius:10px'>
    <h3 style='color:#FF5A5F'>🌟 Our Mission</h3>
    <p>We help Airbnb hosts understand and improve guest satisfaction using advanced analytics.</p>
    
    <h3 style='color:#FF5A5F;margin-top:20px'>🛠️ How It Works</h3>
    <ol>
        <li>Upload your Airbnb listing data</li>
        <li>Our system analyzes key factors</li>
        <li>Get actionable insights to improve</li>
    </ol>
    
    <h3 style='color:#FF5A5F;margin-top:20px'>📊 Data Sources</h3>
    <p>We analyze:</p>
    <ul>
        <li>Property details</li>
        <li>Guest reviews</li>
        <li>Pricing history</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def contact_page():
    st.title("📩 Contact Us")
    
    with st.form("contact_form"):
        name = st.text_input("👤 Your Name")
        email = st.text_input("✉️ Email Address")
        message = st.text_area("💬 Your Message", height=150)
        if st.form_submit_button("📤 Send Message"):
            
            if send_email(name, email, message):
                send_email_to_me(name,"abdoph71@gmail.com",message)
                st.success("✅ Message sent successfully!")
            else:
                st.error("❌ Failed to send message")

# -------------------- Main App --------------------
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Graphs", "About", "Contact"],
            icons=["house", "bar-chart", "info-circle", "envelope"],
            styles={
                "container": {"padding": "5px"},
                "icon": {"color": "#FF5A5F", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "margin": "5px"},
                "nav-link-selected": {"background-color": "#FF5A5F"}
            }
        )
    
    if selected == "Home":
        home_page()
    elif selected == "Graphs":
        graphs_page()
    elif selected == "About":
        about_page()
    elif selected == "Contact":
        contact_page()

if __name__ == "__main__":
    main()