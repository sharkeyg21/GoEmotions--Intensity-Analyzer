
# SECTION 1: IMPORTS, MAPPINGS, AND HELPER FUNCTIONS

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# GoEmotions numerical IDs to string names
id_to_emotion = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement", 14: "fear",
    15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

# Emotion strings directly to their respective intensity tiers
id_to_tier = {
    "annoyance": "Low Intensity", "approval": "Low Intensity", "caring": "Low Intensity", 
    "confusion": "Low Intensity", "curiosity": "Low Intensity", "realization": "Low Intensity",
    "admiration": "Medium Intensity", "amusement": "Medium Intensity", "anger": "Medium Intensity", 
    "desire": "Medium Intensity", "disappointment": "Medium Intensity", "disapproval": "Medium Intensity", 
    "disgust": "Medium Intensity", "embarrassment": "Medium Intensity", "gratitude": "Medium Intensity", 
    "optimism": "Medium Intensity", "remorse": "Medium Intensity", "sadness": "Medium Intensity", 
    "surprise": "Medium Intensity",
    "excitement": "High Intensity", "fear": "High Intensity", "grief": "High Intensity", 
    "joy": "High Intensity", "love": "High Intensity", "nervousness": "High Intensity", 
    "pride": "High Intensity", "relief": "High Intensity", "neutral": "Neutral"
}

intensity_mapping = {
    'Low Intensity': ['annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'realization'],
    'Medium Intensity': ['admiration', 'amusement', 'anger', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'gratitude', 'optimism', 'remorse', 'sadness', 'surprise'],
    'High Intensity': ['excitement', 'fear', 'grief', 'joy', 'love', 'nervousness', 'pride', 'relief']
}

def evaluate_by_intensity(y_true, y_pred, id_to_tier):
    # FIXED: Added 'Neutral' to prevent KeyError crashes!
    intensity_scores = {
        'Low Intensity': [],
        'Medium Intensity': [],
        'High Intensity': [],
        'Neutral': []
    }
    
    emotion_to_tier = {}
    for tier_name, emotions in intensity_mapping.items():
        for emo in emotions:
            emotion_to_tier[emo] = tier_name
                
    for true, pred in zip(y_true, y_pred):
        true_clean = str(true).strip()
        
        if true_clean in id_to_tier:
            tier = id_to_tier[true_clean]
        elif true in id_to_tier:
            tier = id_to_tier[true]
        elif true_clean in emotion_to_tier:
            tier = emotion_to_tier[true_clean]
        else:
            continue
            
        is_correct = (true == pred)
        intensity_scores[tier].append(is_correct)
        
    # FIXED: Added 'Neutral' to the final print loop
    for tier in ['Low Intensity', 'Medium Intensity', 'High Intensity', 'Neutral']:
        if intensity_scores[tier]:
            accuracy = sum(intensity_scores[tier]) / len(intensity_scores[tier])
            print(f"{tier} - Accuracy: {accuracy:.4f} ({len(intensity_scores[tier])} samples)")

# SECTION 2: DATA LOADING & CLEANING

print("Loading data from absolute paths...")
absolute_path_train = "/Users/gracesharkey/Downloads/archive/data/train.tsv" 
absolute_path_test = "/Users/gracesharkey/Downloads/archive/data/test.tsv"

train_raw = pd.read_csv(absolute_path_train, sep="\t", header=None, names=["text", "labels", "id"])
test_raw = pd.read_csv(absolute_path_test, sep="\t", header=None, names=["text", "labels", "id"])

# Filter for single-label sentences
train_single = train_raw[~train_raw['labels'].str.contains(',', na=False)].copy()
test_single = test_raw[~test_raw['labels'].str.contains(',', na=False)].copy()
train_single['labels'] = train_single['labels'].astype(int)
test_single['labels'] = test_single['labels'].astype(int)


# SECTION 3: STRATIFIED TRAIN-TEST SPLIT & LABEL MAPPING

print("\n--- Splitting Data into Train and Validation Sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    train_single['text'], 
    train_single['labels'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_single['labels']
)

y_train_mapped = y_train.map(id_to_emotion)
y_test_mapped = y_test.map(id_to_emotion)


# SECTION 4: TF-IDF VECTORIZATION

print("\n--- Vectorizing Text Features ---")
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# SECTION 5: BASELINE MODEL TRAINING & EVALUATION

print("\n--- Training Baseline Model (Logistic Regression) ---")
baseline = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
baseline.fit(X_train_vec, y_train_mapped)
baseline_preds = baseline.predict(X_test_vec)

print("\n=== Corrected Baseline Classification Report ===")
print(classification_report(y_test_mapped, baseline_preds))
print("\n=== Baseline Performance by Intensity Tiers ===")
evaluate_by_intensity(y_test_mapped, baseline_preds, id_to_tier)

# SECTION 6: ADVANCED MODEL TRAINING (SVM WITH GRID SEARCH) & EVALUATION

print("\n--- Training Tuned SVM Model (Grid Search) ---")
svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
param_grid = {'C': [0.1, 1.0, 10.0]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

svm_grid = GridSearchCV(svm, param_grid, cv=cv, scoring='f1_macro', n_jobs=2)
svm_grid.fit(X_train_vec, y_train_mapped)

best_svm = svm_grid.best_estimator_
svm_preds = best_svm.predict(X_test_vec)

print(f"\nBest SVM Hyperparameter found: C = {svm_grid.best_params_['C']}")
print("\n=== Tuned SVM Classification Report ===")
print(classification_report(y_test_mapped, svm_preds))
print("\n=== SVM Performance by Intensity Tiers ===")
evaluate_by_intensity(y_test_mapped, svm_preds, id_to_tier)



# SECTION 7: EXPORT MODEL ARTIFACTS

print("\n--- Exporting Artifacts for Streamlit UI ---")

# Define the folder path explicitly as a subfolder
export_folder = "/Users/gracesharkey/Desktop/GoEmotions/saved_models"

# Create the subfolder if it doesn't exist
os.makedirs(export_folder, exist_ok=True)

# Define file paths explicitly INSIDE that subfolder
model_file_path = os.path.join(export_folder, "svm_emotion_model.pkl")
vectorizer_file_path = os.path.join(export_folder, "tfidf_vectorizer.pkl")

# Save files explicitly
joblib.dump(best_svm, model_file_path)
joblib.dump(vectorizer, vectorizer_file_path)

print(f"Successfully saved model to: {model_file_path}")
print(f"Successfully saved vectorizer to: {vectorizer_file_path}")