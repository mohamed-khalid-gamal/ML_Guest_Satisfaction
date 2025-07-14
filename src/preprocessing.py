# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn import tree
import pickle
import re
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import os
import string
import warnings
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# Load data
df = pd.read_csv('test_data.csv')
cols_to_remove = [
    'host_acceptance_rate',
    'host_listings_count',
    'square_feet',
    'thumbnail_url',
    'zipcode'
]
df = df.drop([col for col in cols_to_remove if col in df.columns], axis=1)
# Define target and price columns
target_col = 'guest_satisfaction'
price_columns = ['nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee', 'extra_people']

# Clean price columns
for col in price_columns:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100     
with open('scaler_model.pkl', 'rb') as file:      
    scaler = pickle.load(file)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaled_data = scaler.transform(df[num_cols])
with open('imputer_model.pkl', 'rb') as file:
    imputer = pickle.load(file)
imputed_data = imputer.transform(scaled_data)
df[num_cols] = pd.DataFrame(scaler.inverse_transform(imputed_data), columns=num_cols, index=df.index)

# Remove specified columns if they exist

with open('dict_missing_values.pkl', 'rb') as file:
    dict_missing_values = pickle.load(file)

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(dict_missing_values.get(col, 'missing'))
with open('outlier_thresholds.pkl', 'rb') as file:  
    outlier_thresholds = pickle.load(file)
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
with open('scaler_cluster.pkl', 'rb') as file:
    scaler_cluster = pickle.load(file)
cluster_scaled = scaler_cluster.transform(cluster_features)
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)    
df['host_cluster'] = kmeans_model.predict(cluster_scaled)
# Compute price_per_person if accommodates column exists
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
     
# Process text columns
text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview',
             'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about']
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
    print(f"Processing {col}...")
    text_features = df[col].apply(process_text)
    text_features_df = pd.DataFrame(text_features.tolist())
    text_features_df.columns = [f"{col}_{feat}" for feat in text_features_df.columns]
    return text_features_df

def apply_tfidf(df, col, max_features=20):
    MODEL_DIR = "mo"
    print(f"Applying TF-IDF on: {col}")
    try:
        tfidf = joblib.load(os.path.join(MODEL_DIR, f"{col}_tfidf_vectorizer.joblib"))
        tfidf_matrix = tfidf.transform(df[col].fillna(""))
        tfidf_columns = joblib.load(os.path.join(MODEL_DIR, f"{col}_tfidf_columns.joblib"))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns)
        return tfidf_df
    except FileNotFoundError:
        print(f"TF-IDF model for {col} not found. Returning empty DataFrame.")
        return pd.DataFrame()

for col in text_cols:
    if col in df.columns:
        text_features_df = process_column(df, col)
        df = pd.concat([df, text_features_df], axis=1)
        tfidf_df = apply_tfidf(df, col)
        df = pd.concat([df, tfidf_df], axis=1)
        df = df.drop(col, axis=1)

# Compute overall sentiment score
df['overall_sentiment_score'] = (
    df[['summary_sentiment_compound', 'space_sentiment_compound', 'host_about_sentiment_compound']].mean(axis=1)
)

df['price_per_person'] = df['price_per_stay'] / df['accommodates']

# Process date columns

# Clustering
# Map guest_satisfaction
manual_mapping = {'Average': 0, 'High': 1, 'Very High': 2}
if target_col in df.columns:
    df[target_col] = df[target_col].map(manual_mapping)

# Load and apply host_name target encoding
if 'host_name' in df.columns:
    target_means = df.groupby('host_name')[target_col].mean()
    df['host_name_target_encoded'] = df['host_name'].map(target_means).fillna(0)






# Load and apply outlier thresholds

selected_features = []
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
print("coulmns",df.shape)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)  
  
# Prepare X and y
X = df[selected_features]
y = df[target_col]

print("Selected features order:", selected_features)
print("Features in X:", list(X.columns))

# Predictions

models = {
    'logistic': pickle.load(open('logistic.pkl', 'rb')),
    'knn': pickle.load(open('knn1.pkl', 'rb')),
    'decision': pickle.load(open('decision.pkl', 'rb')),
    'svm': pickle.load(open('svm.pkl', 'rb')),
    'randomForest': pickle.load(open('randomForest.pkl', 'rb'))
}
# Import necessary libraries


# Dictionary to store model performance
model_performance = {}
# Function to evaluate a model
def evaluate_model(model, model_name,X,y):
    try:
        # Predict on training and test sets
        y_pred_test = model.predict(X)
        
        # Calculate accuracy
        test_accuracy = accuracy_score(y, y_pred_test)
        
        # Store performance metrics
        model_performance[model_name] = {
            
            'Test Accuracy': test_accuracy
        }
       
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

# 1. Evaluate working models (logistic, svm, randomForest)
working_models = {
    'logistic': models['logistic'],
    'svm': models['svm'],
    'knn' : models['knn'],
    'randomForest': models['randomForest'],
    'decision' : models['decision']
}

for model_name, model in working_models.items():
    evaluate_model(model, model_name, X, y)
evaluate_model(models['randomForest'], "randomForest", X, y)
    
# 5. Performance summary
performance_df = pd.DataFrame(model_performance).T
print("\nModel Performance Summary:")
print(performance_df)
# Plot model performance
plt.figure(figsize=(10, 6))
performance_df[['Test Accuracy']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()