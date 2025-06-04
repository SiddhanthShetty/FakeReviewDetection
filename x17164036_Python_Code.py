#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing packages
import pandas as pd
import numpy as np
import seaborn as sns
import textstat as ts
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
# stop = stopwords.words('english') # spaCy will handle its own stopword list

# spaCy model loading
import spacy
from lightgbm import LGBMClassifier # Import LightGBM
import mlflow # MLflow import
import mlflow.sklearn # MLflow sklearn import
import optuna # Optuna import
from sklearn.pipeline import Pipeline # Pipeline import
from sklearn.preprocessing import StandardScaler # StandardScaler import

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the spaCy POS tagger '
          "(don't worry, this will only happen once)")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Utility functions for model evaluation
def model_report(y_act, y_pred):
    """Prints a classification report and displays AUC-ROC curve."""
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import f1_score, roc_curve,auc
    import matplotlib.pyplot as plt
    print("Confusion Matrix: ")
    print(confusion_matrix(y_act, y_pred))
    print("Accuracy = ", accuracy_score(y_act, y_pred))
    print("Precision = " ,precision_score(y_act, y_pred))
    print("Recall = " ,recall_score(y_act, y_pred))
    print("F1 Score = " ,f1_score(y_act, y_pred))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_act, y_pred)
    print("AUC Score =", auc(false_positive_rate, true_positive_rate))
    print("Kappa score = ",cohen_kappa_score(y_act,y_pred))
    print("Error rate = " ,1 - accuracy_score(y_act, y_pred))
    print("AUC-ROC Curve: ")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(false_positive_rate, true_positive_rate,marker='.')
    plt.show()
    pass

def cross_validation_report(result):
    """Prints the mean accuracy and variance of cross-validation results."""
    print("Mean accuracy: ", result.mean())
    print("Variance: ", result.std())
    pass

def KFold_Cross_Validation(classifier, n, X_train_data, y_train_data):
    """Performs Stratified K-Fold cross-validation and prints the report."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    # Using StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    acc = cross_val_score(estimator = classifier, X = X_train_data, y = y_train_data, cv = skf, scoring = 'accuracy')
    cross_validation_report(acc)
    pass

def load_and_clean_data(reviews_path='data/Reviews.csv', restaurant_path='data/Restaurant.csv'):
    """Loads, cleans, and merges review and restaurant data using relative paths."""
    #Importing Raw Data
    reviews_df = pd.read_csv(reviews_path, encoding = 'cp1252')
    restaurant_df = pd.read_csv(restaurant_path, encoding = 'cp1252')

    #Checking for Missing values - Reviews Dataset
    column_names = reviews_df.columns
    print(column_names) #Column names
    totalCells = np.product(reviews_df.shape) #Calculate total number of cells in dataframe
    missingCount = reviews_df.isnull().sum() #Count number of missing values per column
    totalMissing = missingCount.sum() #Calculate total number of missing values
    print("The Reviews dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.") #Calculate percentage of missing values

    #Replace any missing values with NA and Drop NA values
    reviews_df.replace(' ',np.nan, inplace = True)
    reviews_df = reviews_df.dropna()

    #Deleting rows flagged as NR and YR
    reviews_df = reviews_df[reviews_df.flagged != 'NR']
    reviews_df = reviews_df[reviews_df.flagged != 'YR']

    #Encoding categorical values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    reviews_df['flagged_code'] = le.fit_transform(reviews_df['flagged'])

    #Unique Id column
    reviews_df['ReviewID'] = reviews_df.index

    #Adding review content wordcount feature
    reviews_df['WordCount_Review'] = reviews_df['reviewContent'].apply(lambda comment: len(comment.split()))

    #Drop unwanted columns
    reviews_df = reviews_df.drop(columns=['date','reviewID','coolCount','funnyCount'], axis = 1)

    #Renaming Columns
    reviews_df.rename(columns={'rating':'ReviewRating','usefulCount':'UsefulCount_Review'}, inplace = True)

    #Renaming Columns - Restaurant dataset
    restaurant_df.rename(columns={'reviewCount':'TotalReviewCountofRestaurant','filReviewCount':'FakeReviewCountRestaurant','rating':'AggRestaurantRating'}, inplace = True)
    rest_df = restaurant_df[['restaurantID','TotalReviewCountofRestaurant','FakeReviewCountRestaurant','AggRestaurantRating']]

    #Joining the two datasets
    combined_df = pd.merge(reviews_df,rest_df, how = 'inner', on = 'restaurantID')
    combined_df = pd.DataFrame(combined_df)
    # combined_df.to_csv(r'D:\\NCI Notes\\Thesis\\Data\\FinalMergedDataset.csv') # This will be done outside the function
    return combined_df

# In[2]:


# Call the function to load and clean data
combined_df = load_and_clean_data() # Using default relative paths
combined_df.to_csv('data/FinalMergedDataset.csv')

def preprocess_text_features(df, text_column='reviewContent', nlp_model=nlp):
    """Preprocessing text features in a DataFrame using spaCy."""

    # Remove numbers first (spaCy doesn't remove them by default), and convert to lowercase
    # spaCy's lemmatizer works better on original casing, but for consistency with previous lowercasing, apply here.
    # Alternatively, ensure tokens are lowercased after lemmatization if not already.
    df['reviewContentCleaned'] = df[text_column].apply(lambda x: "".join([i for i in x if not i.isdigit()]).lower())

    processed_texts = []
    stopword_counts = []

    # Process documents using spaCy's pipeline for efficiency
    for doc in nlp_model.pipe(df['reviewContentCleaned'], disable=["parser", "ner"]):
        # For 'reviewContentNew': lemmatize and remove stopwords and punctuation
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        processed_texts.append(" ".join(lemmatized_tokens))

        # For 'stopwordsCount': count tokens that are identified as stopwords by spaCy in the (number-removed, lowercased) text
        current_stopwords_count = sum(1 for token in doc if token.is_stop)
        stopword_counts.append(current_stopwords_count)

    df['reviewContentNew'] = processed_texts
    df['stopwordsCount'] = stopword_counts

    # Drop the intermediate column if no longer needed
    df.drop(columns=['reviewContentCleaned'], inplace=True)

    return df

# In[10]:


combined_df.head()


# In[11]:


combined_df.info()


# In[12]:


combined_df.describe()


# In[13]:


# Preprocess text features
combined_df = preprocess_text_features(combined_df)

def engineer_features(df, text_column='reviewContent'):
    """Engineers numeric and readability features from text."""
    #Adding new numeric features - Feature Engineering
    #Rating features and Text based statistics
    #AggRating Deviation
    df['DeviationfromAggRating'] = abs(df['AggRestaurantRating'] - df['ReviewRating'])
    #Character count
    df['charCount'] = df[text_column].apply(len)
    #uppercase count
    df['uppercaseCount'] = df[text_column].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    #special char count
    df['specialCharCount'] = df[text_column].apply(lambda comment: sum (comment.count(w) for w in '[\w]+'))
    #sentence count
    df['sentenceCount'] = df[text_column].apply(ts.sentence_count)

    #Redability features
    df['fleschReadingEase'] = df[text_column].apply(ts.flesch_reading_ease)
    df['fleschKincaidGrade'] = df[text_column].apply(ts.flesch_kincaid_grade)
    df['fogScale'] = df[text_column].apply(ts.gunning_fog)
    df['smogScore'] = df[text_column].apply(ts.smog_index)
    df['ARI'] = df[text_column].apply(ts.automated_readability_index)
    df['CLI'] = df[text_column].apply(ts.coleman_liau_index)
    df['linsearWrite'] = df[text_column].apply(ts.linsear_write_formula)
    df['daleChallScore'] = df[text_column].apply(ts.dale_chall_readability_score)

    #Sentiment Score
    #df['sentimentScore'] = df[text_column].apply(lambda x: TextBlob(x).sentiment[0])#SentimentScore-polarity
    return df

# In[14]:


combined_df['reviewContentNew'].head()


# In[15]:


# Engineer features
combined_df = engineer_features(combined_df)

def create_pos_features(df, text_column='reviewContent'):
    """Creates POS tagging features from text."""
    from nltk import word_tokenize, pos_tag # Ensure nltk is imported here
    def count_noun(text):
        nouns = sum(1 for word, pos in pos_tag(word_tokenize(text)) if pos.startswith('NN'))
        return nouns
    def count_verb(text):
        verbs = sum(1 for word, pos in pos_tag(word_tokenize(text)) if pos.startswith('VB'))
        return verbs
    def count_adjective(text):
        adj = sum(1 for word, pos in pos_tag(word_tokenize(text)) if pos.startswith('JJ'))
        return adj
    def count_adverb(text):
        adv = sum(1 for word, pos in pos_tag(word_tokenize(text)) if pos.startswith('RB'))
        return adv

    df['nounCount'] = df[text_column].apply(count_noun)
    df['verbCount'] = df[text_column].apply(count_verb)
    df['adjectiveCount'] = df[text_column].apply(count_adjective)
    df['adverbCount'] = df[text_column].apply(count_adverb)
    return df

# In[18]:


# Create POS features
combined_df = create_pos_features(combined_df)

# Placeholder for word embedding generation
def generate_word_embeddings(df, text_column_name='reviewContentNew'):
    """
    Placeholder function for generating word embeddings.
    This function is not fully implemented and serves as a marker for future enhancement.
    """
    # Option 1: Using spaCy's built-in word vectors
    # ------------------------------------------------
    # For this to work, a larger spaCy model with vectors is needed,
    # e.g., 'en_core_web_md' or 'en_core_web_lg'.
    # The current 'en_core_web_sm' does not include static vectors by default.
    # If a suitable model is loaded (e.g., nlp_lg = spacy.load('en_core_web_lg')),
    # document vectors can be obtained directly from spaCy Doc objects.
    # These are often derived by averaging token vectors if `doc.has_vector` is true.

    # Example (conceptual):
    # if nlp.meta['vectors']['width'] > 0:  # Check if the loaded nlp object has vectors
    #     df['doc_vector'] = df[text_column_name].apply(lambda text: nlp(text).vector)
    # else:
    #     print(f"The current spaCy model ('{nlp.meta['name']}') does not have built-in word vectors. "
    #           "Consider using 'en_core_web_md' or 'en_core_web_lg'.")
    #     # As a fallback or alternative, one could average vectors of individual tokens if they exist:
    #     # def get_avg_vector(text, nlp_model):
    #     #     doc = nlp_model(text)
    #     #     # Filter out OOV tokens and tokens without vectors
    #     #     vectors = [token.vector for token in doc if token.has_vector and not token.is_oov]
    #     #     if vectors:
    #     #         return np.mean(vectors, axis=0)
    #     #     else:
    #     #         return np.zeros(nlp_model.meta.get('vectors', {}).get('width', 300)) # Return zero vector of appropriate size
    #     # df['doc_vector'] = df[text_column_name].apply(lambda text: get_avg_vector(text, nlp)) # Assuming nlp is the loaded spaCy model

    # Option 2: Integrating pre-trained Word2Vec, GloVe, or FastText models
    # ---------------------------------------------------------------------
    # Libraries like `gensim` can be used to load pre-trained models.
    # import gensim.downloader as api
    # wv_model = api.load('word2vec-google-news-300') # Example: Word2Vec
    # glove_model = api.load('glove-wiki-gigaword-100') # Example: GloVe
    # fasttext_model = api.load('fasttext-wiki-news-subwords-300') # Example: FastText

    # def document_vector_from_pretrained(text, model):
    #     words = text.split()
    #     word_vectors = [model[word] for word in words if word in model]
    #     if not word_vectors:
    #         return np.zeros(model.vector_size) # Or model.wv.vector_size for some gensim models
    #     return np.mean(word_vectors, axis=0)

    # df['word2vec_embedding'] = df[text_column_name].apply(lambda text: document_vector_from_pretrained(text, wv_model))

    # Output:
    # --------
    # The function would typically add new columns to the DataFrame, where each column
    # (or a set of columns) represents the document embedding.
    # For instance, if embeddings are 300-dimensional, one might have a single column
    # containing a NumPy array, or 300 separate columns.
    # These embeddings can then be used as features for machine learning models.

    print("Placeholder function `generate_word_embeddings` called. "
          "This is where word embedding generation logic would be implemented.")

    # For now, it doesn't modify the DataFrame
    return df

# In[19]:


combined_df.info()


# In[20]:


#Exporting the PreProcessed Data
combined_df.to_csv('data/CombinedPreProcessedDataset.csv')

# In[22]:


# Future enhancement: Call generate_word_embeddings(combined_df) here if implemented.
# combined_df = generate_word_embeddings(combined_df, 'reviewContentNew')

#Checking class balance in the dataset
combined_df['flagged'].value_counts()


# In[23]:


combined_df['flagged_code'].value_counts()


# In[24]:


import seaborn as sns
sns.countplot(x = 'flagged', data = combined_df)


# In[25]:


#Handling Class Imbalance
#Random Undersampling for ngrams model
min_class_length = len(combined_df[combined_df['flagged_code'] == 1])
maj_class_indices = combined_df[combined_df['flagged_code'] == 0].index
random_maj_class_indices = np.random.choice(maj_class_indices,min_class_length,replace = False)
min_class_indices = combined_df[combined_df['flagged_code'] == 1].index
random_usamp_indices = np.concatenate([min_class_indices,random_maj_class_indices])
text_df = combined_df.loc[random_usamp_indices]


# In[26]:


text_df.info()


# In[27]:


import seaborn as sns
sns.countplot(x = 'flagged', data = text_df)


# In[28]:


#Export Random Undersampled Dataset to csv
text_df.to_csv('data/UndersampledDataset.csv')


# In[29]:


#Handling Class Imbalance & Model Training for Numeric Features

# Define X and y for numeric features (excluding readability for this first section)
numeric_df = combined_df.drop(columns=['reviewerID','reviewContent','flagged','ReviewID','restaurantID','reviewContentNew'])
X_numeric_original = numeric_df.drop(columns=['flagged_code','fleschReadingEase',
       'fleschKincaidGrade', 'fogScale', 'smogScore', 'ARI', 'CLI',
       'linsearWrite', 'daleChallScore'])
# X_numeric_original_values = X_numeric_original.iloc[:,:].values # Keep as DataFrame for column names if Boruta is used later
y_numeric_original = numeric_df['flagged_code'].values

# Split original numeric data before any preprocessing like SMOTE or Boruta for pipeline
from sklearn.model_selection import train_test_split
X_train_num_orig, X_test_num_orig, y_train_num_orig, y_test_num_orig = train_test_split(
    X_numeric_original, y_numeric_original, test_size=0.20, random_state=123, stratify=y_numeric_original
)


# BorutaPy Feature Selection (Commentary)
# ---------------------------------------
# BorutaPy is typically used to select features. In the original script, it was applied *after* SMOTE.
# If integrating Boruta into a pipeline with SMOTE, care must be taken:
# 1. `imblearn.pipeline.Pipeline` should be used if SMOTE is a step, as it handles sampler application correctly during cross-validation.
# 2. BorutaTransformer could be created, or feature selection could happen as a preliminary step on the (potentially SMOTE'd) training data.
# For this current refactoring, we will proceed with training some models *without* Boruta first to demonstrate the pipeline,
# and then re-introduce Boruta-selected features for other models or as a separate step.
# The features selected by Boruta in the original script (feat_selector.support_) were based on SMOTE'd data.
# We will first demonstrate a pipeline on X_train_num_orig, y_train_num_orig.
# Then, we will re-apply SMOTE and Boruta as per the original script flow for other models to maintain consistency with their original setup.

# --- Pipeline for XGBoost with Optuna-tuned parameters ---
# The Optuna study `study_xgb_numeric` was defined earlier (around original In[37]).
# It was trained on data that had undergone SMOTE and Boruta selection.
# Ideally, for a pipeline that includes SMOTE, the hyperparameter tuning
# should be done on the pipeline itself or on data reflecting only pre-SMOTE preprocessing (e.g., scaling).
# Using `study_xgb_numeric.best_params` here is a pragmatic choice based on availability,
# but for optimal performance, a new Optuna study dedicated to this pipeline configuration would be better.
# This note highlights a potential area for future refinement.

if 'study_xgb_numeric' not in globals() or not hasattr(study_xgb_numeric, 'best_params'):
    print("Warning: Optuna study 'study_xgb_numeric' or its best_params not found. Using default XGBoost params for pipeline.")
    optuna_best_params_for_pipeline = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5} # Sensible defaults
else:
    optuna_best_params_for_pipeline = study_xgb_numeric.best_params
    print(f"Using Optuna parameters from 'study_xgb_numeric' for the pipeline: {optuna_best_params_for_pipeline}")


# Define the pipeline
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb.XGBClassifier(**optuna_best_params_for_pipeline,
                                     random_state=42, objective='binary:logistic',
                                     eval_metric='logloss', use_label_encoder=False))
])

# Train and evaluate the pipeline with MLflow
with mlflow.start_run(run_name="XGBoost_Numeric_Pipeline_with_Prior_Optuna_Params"): # Clarified run name
    # Log pipeline structure and classifier params used
    pipeline_params_to_log = {}
    for step_name, step_obj in xgb_pipeline.steps:
        pipeline_params_to_log[f"{step_name}_class"] = step_obj.__class__.__name__
        if hasattr(step_obj, 'get_params'):
            for k,v in step_obj.get_params(deep=False).items(): # Log non-default params of steps
                 pipeline_params_to_log[f"{step_name}_{k}"] = v

    # Overwrite classifier params with the ones from Optuna explicitly for clarity in logs
    for k,v in optuna_best_params_for_pipeline.items():
        pipeline_params_to_log[f"classifier_{k}"] = v

    mlflow.log_params(pipeline_params_to_log)

    xgb_pipeline.fit(X_train_num_orig, y_train_num_orig)

    pred_y_pipeline = xgb_pipeline.predict(X_test_num_orig)

    pipeline_metrics = model_report(y_test_num_orig, pred_y_pipeline, model_name="XGBoost Numeric Pipeline (Optuna Tuned)")
    mlflow.log_metrics(pipeline_metrics)

    mlflow.sklearn.log_model(xgb_pipeline, "xgboost-numeric-pipeline-optuna-tuned")
    print(f"MLflow Run ID for XGBoost Numeric Pipeline: {mlflow.active_run().info.run_id}")

# The KFold_Cross_Validation for this pipeline would involve cloning the pipeline for each fold.
# This is often handled by Scikit-learn's cross_val_score directly with the pipeline object.
# For simplicity, we'll demonstrate it here if needed, or rely on previous CV for individual models.
# print("\nPerforming K-Fold Cross Validation for the XGBoost Pipeline:")
# KFold_Cross_Validation(xgb_pipeline, 5, X_train_num_orig, y_train_num_orig) # Reduced splits for pipeline CV speed


# --- Re-establishing SMOTE and Boruta for other models in this section (to follow original script logic) ---
# The following steps are to ensure that X_train, y_train, X_test, X (for Boruta) are prepared
# as they were in the original script for the subsequent non-pipeline models in this section.

print("\nRe-applying SMOTE and Boruta for subsequent non-pipeline models in this section...")
sm_orig = SMOTE(random_state=42) # Renamed to avoid conflict if sm is used elsewhere
X_smoted_for_boruta, y_smoted_for_boruta = sm_orig.fit_resample(X_numeric_original, y_numeric_original)

# Performing feature selection on numeric features using BorutaPy (as in original script)
# This is on the SMOTE'd data of the *entire* numeric dataset (before train/test split for Boruta's fit)
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc_boruta = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
feat_selector_numeric = BorutaPy(rfc_boruta, n_estimators='auto', verbose=0, random_state=0) # verbose to 0
print("Fitting BorutaPy for numeric features (this might take a while)...")
feat_selector_numeric.fit(X_smoted_for_boruta.values if isinstance(X_smoted_for_boruta, pd.DataFrame) else X_smoted_for_boruta, y_smoted_for_boruta)
print("BorutaPy fitting complete.")

# Updating feature set for training with selected features
X_boruta_selected_features = X_smoted_for_boruta.iloc[:, feat_selector_numeric.support_] if isinstance(X_smoted_for_boruta, pd.DataFrame) else X_smoted_for_boruta[:, feat_selector_numeric.support_]

# Now, split the Boruta selected and SMOTE'd data for the other models
X_train, X_test, y_train, y_test = train_test_split(
    X_boruta_selected_features,
    y_smoted_for_boruta, # y corresponds to the SMOTE'd X
    test_size=0.20,
    random_state=123,
    stratify=y_smoted_for_boruta # Stratify on the resampled y
)
# The global X, y were also used by Boruta. For clarity, let's rename the Boruta input:
# X_for_boruta_fit = X_smoted_for_boruta.values if isinstance(X_smoted_for_boruta, pd.DataFrame) else X_smoted_for_boruta
# y_for_boruta_fit = y_smoted_for_boruta

# The rest of the models in this section (GaussianNB, original XGBoost, AdaBoost, GradientBoosting, LightGBM)
# will use X_train, y_train, X_test, y_test derived from SMOTE + Boruta selected features.
# The Optuna-tuned XGBoost that was previously here is now replaced by the pipeline version above for this specific model.
# The MLflow run for "XGBoost_Numeric_Optuna_Tuned" is now effectively the pipeline run.
# We need to ensure the non-pipeline XGBoost (untuned and Optuna-tuned) are distinct or one is removed.
# The original untuned XGBoost (cell In[36]) and Optuna-tuned (cell In[40]) were on Boruta selected features.
# The new pipeline XGBoost is *not* on Boruta selected features from this specific Boruta run.

# For clarity, the Optuna-tuned XGBoost (non-pipeline) will be removed from this section,
# as its new version is the pipeline. The untuned XGBoost (In[36]) can remain for comparison.

# In[31]:


#Class instances after upsampling using SMOTE
print("Classes after SMOTE : ")
print("Count of label '1' = {}".format(sum(y==1)))
print("Count of label '0' = {}".format(sum(y==0)))


# In[32]:
# This Boruta fitting was moved up to operate on X_smoted_for_boruta, y_smoted_for_boruta
# and its output X_boruta_selected_features is then split into X_train, X_test, y_train, y_test.
# The feat_selector object is now feat_selector_numeric.

# from sklearn.ensemble import RandomForestClassifier # Already imported
# from boruta import BorutaPy # Already imported
# rfc = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
# feat_selector = BorutaPy(rfc, n_estimators='auto',verbose=2, random_state=0)
# feat_selector.fit(X,y) # X and y here were SMOTE'd version of original X,y


# In[33]:

# This transformation was also part of the moved Boruta block.
# X = X[:,feat_selector.support_]


# In[34]:

# This train_test_split is now done above, using X_boruta_selected_features and y_smoted_for_boruta.
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 123)


# In[35]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_y = gnb.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gnb,10, X_train, y_train)


# In[36]:


#XGBoost without parameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[37]:


# Hyperparameter Tuning for XGBoost using Optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective_xgb(trial, X, y):
    # Define search space
    # Ranges are inspired by the original RandomizedSearchCV, adjusted for Optuna
    param = {
        'verbosity': 0, # Suppress XGBoost messages
        'objective': 'binary:logistic',
        'eval_metric': 'auc', # Using AUC for optimization is common
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200), # Adjusted range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        # 'eta' is an alias for learning_rate in XGBoost, avoid using both
        'random_state': 42
    }
    # For 'dart' booster, suggest rate_drop and skip_drop if it's chosen
    if param['booster'] == 'dart':
        param['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5, step=0.1)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5, step=0.1)

    model = xgb.XGBClassifier(**param)

    # Using StratifiedKFold for cross-validation as in KFold_Cross_Validation function
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5 splits for faster trial, can be increased
    # Using accuracy as the metric to maximize, consistent with original RandomizedSearchCV scoring
    # Note: KFold_Cross_Validation prints mean and std, here we need the mean for Optuna
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return scores.mean()

# Create a study object and optimize
# Note: xgbc (untuned XGBoost) was defined in In[36] - it's okay as we instantiate new models here
study_xgb_numeric = optuna.create_study(direction='maximize', study_name='XGBoost_Numeric_Optuna')
study_xgb_numeric.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=30) # Reduced n_trials for speed in this example

print("Best trial for XGBoost (Numeric Features):")
print(f"  Value (Accuracy): {study_xgb_numeric.best_value:.4f}")
print("  Params: ")
for key, value in study_xgb_numeric.best_params.items():
    print(f"    {key}: {value}")

# Commenting out old RandomizedSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# parameter = {
#         'max_depth' : np.arange(3,10,1),
#         'n_estimators' : range(1,100,1),
#         'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60],
#         'eta' : range(1,100,1), # eta is an alias for learning_rate
#         'subsample' : np.arange(0.5,1,0.01),
#         'min_child_weight': range(1,6,1),
#         'gamma' : [i/10.0 for i in range(0,5)]
#         }
# rs = RandomizedSearchCV(
#         estimator = xgbc, # xgbc here was the untuned one from In[36]
#         param_distributions = parameter,
#         n_iter = 20,
#         scoring ='accuracy',
#         n_jobs=4,
#         verbose=10,
#         random_state=10
#         )
# rs.fit(X_train,y_train)

# In[38]:
# rs.cv_results_ # Commented out as rs is no longer used

# In[39]:
# print("Best accuracy Obtained: {0}".format(rs.best_score_)) # Commented out
# print("Best Parameters: ") # Commented out
# for key, value in rs.best_params_.items(): # Commented out
#     print("\t{}:{}".format(key, value)) # Commented out


# In[40]:


#XGBoost after Parameter tuning with Optuna
with mlflow.start_run(run_name="XGBoost_Numeric_Optuna_Tuned"): # Updated run name
    import xgboost as xgb

    tuned_params_optuna = study_xgb_numeric.best_params
    mlflow.log_params(tuned_params_optuna)
    # Ensure all necessary parameters for XGBClassifier are included, even if not tuned by Optuna but required.
    # e.g. objective, eval_metric if not default, random_state
    xgbc_optuna_tuned = xgb.XGBClassifier(**tuned_params_optuna, random_state=42,
                                          objective='binary:logistic', eval_metric='auc') # Added objective & eval_metric for consistency

    xgbc_optuna_tuned.fit(X_train,y_train)
    pred_y = xgbc_optuna_tuned.predict(X_test)

    metrics = model_report(y_test,pred_y, model_name="XGBoost Optuna Tuned (Numeric Features)")
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(xgbc_optuna_tuned, "xgboost-numeric-optuna-tuned-model")
    print(f"MLflow Run ID for Optuna-tuned XGBoost: {mlflow.active_run().info.run_id}")

KFold_Cross_Validation(xgbc_optuna_tuned,10, X_train, y_train) # CV is outside the main run for this single split


# In[41]:


#AdaBoost without Parameter tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[42]:


#Hyperparameter Tuning for AdaBoost
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0]
        }

rs = RandomizedSearchCV(
        estimator = abc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[43]:


rs.cv_results_


# In[44]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Best Parameters:")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[45]:


#After parameter Tuning - AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier(n_estimators = 72,
                         learning_rate = 0.9)
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[46]:


#Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# In[47]:


#HyperParameter Tuning for Gradient Boosting Machine
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0],
        'subsample' : np.arange(0.5,1,0.01),
        'min_samples_split' : range(50,500,50),
        'min_samples_leaf' : range(10,400,20),
        'max_depth' : np.arange(3,10,1)
        
        }

rs = RandomizedSearchCV(
        estimator = gbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[48]:


rs.cv_results_


# In[49]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[50]:


#After parameter Tuning Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(subsample = 0.7900000000000003,
        n_estimators = 79,
        min_samples_split = 300,
        min_samples_leaf = 10,
        max_depth = 8,
        learning_rate = 0.15)
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train,y_train)
pred_y = lgbm.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(lgbm,10, X_train, y_train)


# In[51]:


#Adding Readability features in the feature set
X = combined_df.drop(columns=['reviewerID','reviewContent','flagged','flagged_code','ReviewID','restaurantID',
                              'reviewContentNew'])
X = X.iloc[:,:].values
y = combined_df.iloc[:,combined_df.columns.get_loc('flagged_code')].values


# In[52]:


#Class instances after upsampling using SMOTE
print("Classes before SMOTE : ")
print("Count of label '1' = {}".format(sum(y==1)))
print("Count of label '0' = {}".format(sum(y==0)))


# In[53]:


#Handling Class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_sample(X, y)


# In[54]:


#Class instances after upsampling using SMOTE
print("Classes after SMOTE : ")
print("Count of label '1' = {}".format(sum(y==1)))
print("Count of label '0' = {}".format(sum(y==0)))


# In[55]:


#Feature Selection using Boruta Py
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
feat_selector = BorutaPy(rfc, n_estimators='auto',verbose=2, random_state=0)
feat_selector.fit(X,y)


# In[56]:


X = X[:,feat_selector.support_]


# In[57]:


#Split Dataset for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 123)


# In[58]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_y = gnb.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gnb,10, X_train, y_train)


# In[59]:


#XGBoost before Hyperparameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[60]:


#Parameter Tuning
from sklearn.model_selection import RandomizedSearchCV

parameter = {   
        'max_depth' : np.arange(3,10,1),
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60],
        'eta' : range(1,100,1),
        'subsample' : np.arange(0.5,1,0.01),
        'min_child_weight': range(1,6,1),
        'gamma' : [i/10.0 for i in range(0,5)]
        }

rs = RandomizedSearchCV(
        estimator = xgbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[61]:


rs.cv_results_


# In[62]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[63]:


#XGBoost after Hyperparameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier(subsample = 0.8500000000000003,
        n_estimators = 74,
        min_child_weight = 2,
        max_depth = 8,
        learning_rate = 0.3,
        gamma = 0.1,
        eta = 54)
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[64]:


#AdaBoost without parameter Tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[65]:


#Parameter Tuning - AdaBoost
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0]
        }

rs = RandomizedSearchCV(
        estimator = abc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[66]:


rs.cv_results_


# In[67]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[68]:


#AdaBoost after tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier(n_estimators = 72,
                         learning_rate = 0.9)
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[69]:


#Gradient Boosting Machine without parameter tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# In[70]:


#Parameter Tuning for Gradient Boosting Machine
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0],
        'subsample' : np.arange(0.5,1,0.01),
        'min_samples_split' : range(50,500,50),
        'min_samples_leaf' : range(10,400,20),
        'max_depth' : np.arange(3,10,1)
        }

rs = RandomizedSearchCV(
        estimator = gbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[71]:


rs.cv_results_


# In[72]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[73]:


#Gradient Boosting after Parameter Tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier( subsample = 0.7900000000000003,
        n_estimators = 79,
        min_samples_split = 300,
        min_samples_leaf = 10,
        max_depth = 8,
        learning_rate = 0.15)
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train,y_train)
pred_y = lgbm.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(lgbm,10, X_train, y_train)


# In[124]:


#Bi-Grams TFIDF and numeric features Model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range=(2,2), analyzer='word', min_df = 5)
tv.fit(text_df['reviewContentNew'])
tfidf_bigram_df = tv.fit_transform(text_df['reviewContentNew'])
tfidf_bigram_df = tfidf_bigram_df.toarray()
tfidf_bigram_df = pd.DataFrame(tfidf_bigram_df,columns=tv.get_feature_names())
tfidf_bigram_df = tfidf_bigram_df.reset_index(drop=True)
text_df = text_df.reset_index(drop=True)
combined_bigram_df = pd.concat([text_df,tfidf_bigram_df], axis=1)


# In[125]:


combined_bigram_df.head()


# In[126]:


#Dropping unwanted columns
combined_bigram_df = combined_bigram_df.drop(columns=['reviewerID','reviewContent','flagged','restaurantID','ReviewID',
                                                      'reviewContentNew','sentimentScore'])


# In[131]:


combined_bigram_df = combined_bigram_df.abs()


# In[132]:


#Splitting target varibles and independent variables
X = combined_bigram_df.drop(columns=['flagged_code'])
X = X.iloc[:,:].values
y = combined_bigram_df.iloc[:,combined_bigram_df.columns.get_loc('flagged_code')].values


# In[133]:


X.shape


# In[134]:


y.shape


# In[135]:


#Feature selection by using Chi-Square test
#Chi Square feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi_feature = SelectKBest(chi2, k=4000)
X_KBest = chi_feature.fit_transform(X, y)


# In[136]:


#Splitting dataset into train and test after feature selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_KBest,y,test_size = 0.20,random_state = 123)


# In[137]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_y = gnb.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gnb,10, X_train, y_train)


# In[84]:


#XGBoost before HyperParameter tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[85]:


#Parameter Tuning for XGBoost
from sklearn.model_selection import RandomizedSearchCV

parameter = {   
        'max_depth' : np.arange(3,10,1),
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90],
        'eta' : range(1,100,1),
        'subsample' : np.arange(0.5,1,0.01),
        'min_child_weight': range(1,6,1),
        'gamma' : [i/10.0 for i in range(0,5)]
        }

rs = RandomizedSearchCV(
        estimator = xgbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[86]:


rs.cv_results_


# In[87]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[88]:


#XGBoost after parameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier(subsample = 0.54,
        n_estimators = 43,
        min_child_weight = 3,
        max_depth = 5,
        learning_rate = 0.1,
        gamma = 0.1,
        eta = 77)
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y) 
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[89]:


#AdaBoost before parameter tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[90]:


#Parameter Tuning AdaBoost
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0]
        }

rs = RandomizedSearchCV(
        estimator = abc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[91]:


rs.cv_results_


# In[92]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[93]:


#After parameter Tuning - AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier(n_estimators = 77,
                         learning_rate = 0.3)
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[94]:


#Gradient Boosting Machine before tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# In[95]:


#Parameter Tuning GradientBoosting
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0],
        'subsample' : np.arange(0.5,1,0.01),
        'min_samples_split' : range(50,500,50),
        'min_samples_leaf' : range(10,400,20),
        'max_depth' : np.arange(3,10,1)
        
        }

rs = RandomizedSearchCV(
        estimator = gbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[96]:


rs.cv_results_


# In[97]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[98]:


#Gradient Boosting Machine after tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(subsample = 0.7900000000000003,
        n_estimators = 79,
        min_samples_split = 300,
        min_samples_leaf = 10,
        max_depth = 8,
        learning_rate = 0.15)
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train,y_train)
pred_y = lgbm.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(lgbm,10, X_train, y_train)


# In[99]:


#TFIDF Trigram model
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range=(3,3), analyzer='word', min_df = 2)
tv.fit(text_df['reviewContentNew'])
tfidf_trigram_df = tv.fit_transform(text_df['reviewContentNew'])
tfidf_trigram_df = tfidf_trigram_df.toarray()
tfidf_trigram_df = pd.DataFrame(tfidf_trigram_df,columns=tv.get_feature_names())
tfidf_trigram_df = tfidf_trigram_df.reset_index(drop=True)
text_df = text_df.reset_index(drop=True)
combined_trigram_df = pd.concat([text_df,tfidf_trigram_df], axis=1)


# In[100]:


#Dropping unwanted columns
combined_trigram_df = combined_trigram_df.drop(columns=['reviewerID','reviewContent','flagged','restaurantID','ReviewID',
                                                      'reviewContentNew','sentimentScore'])


# In[101]:


combined_trigram_df = combined_trigram_df.abs()


# In[102]:


#Splitting target varibles and independent variables
X = combined_trigram_df.drop(columns=['flagged_code'])
X = X.iloc[:,:].values
y = combined_trigram_df.iloc[:,combined_trigram_df.columns.get_loc('flagged_code')].values


# In[103]:


X.shape


# In[104]:


y.shape


# In[105]:


#Feature selection using Chi-Square test
#Chi Square feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi_feature = SelectKBest(chi2, k=3000)
X_KBest = chi_feature.fit_transform(X, y)


# In[106]:


#Splitting after feature selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_KBest,y,test_size = 0.20,random_state = 123)


# In[107]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_y = gnb.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gnb,10, X_train, y_train)


# In[108]:


#XGBoost before HyperParameter tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[109]:


#Parameter Tuning - XGBoost
from sklearn.model_selection import RandomizedSearchCV

parameter = {   
        'max_depth' : np.arange(3,10,1),
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60],
        'eta' : range(1,100,1),
        'subsample' : np.arange(0.5,1,0.01),
        'min_child_weight': range(1,6,1),
        'gamma' : [i/10.0 for i in range(0,5)]
        }

rs = RandomizedSearchCV(
        estimator = xgbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[110]:


rs.cv_results_


# In[111]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[112]:


#XGBoost After Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier(subsample = 0.8400000000000003,
        n_estimators = 58,
        min_child_weight = 3,
        max_depth = 4,
        learning_rate = 0.1,
        gamma = 0.4,
        eta = 83)
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y) 
KFold_Cross_Validation(xgbc,10, X_train, y_train)


# In[113]:


#AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[114]:


#Parameter Tuning AdaBoost
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0]
        }

rs = RandomizedSearchCV(
        estimator = abc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[115]:


rs.cv_results_


# In[116]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[117]:


#After parameter Tuning - AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier(n_estimators = 98,
                         learning_rate = 0.6)
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10, X_train, y_train)


# In[118]:


#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# In[119]:


#Parameter Tuning Gradient Boosting Machine
parameter = {
        'n_estimators' : range(1,100,1),
        'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50,0.60,0.70,0.80,0.90,1.0],
        'subsample' : np.arange(0.5,1,0.01),
        'min_samples_split' : range(50,500,50),
        'min_samples_leaf' : range(10,400,20),
        'max_depth' : np.arange(3,10,1)
        
        }

rs = RandomizedSearchCV(
        estimator = gbc,
        param_distributions = parameter,
        n_iter = 20,
        scoring ='accuracy',
        n_jobs=4,
        verbose=10,
        random_state=10
        )

rs.fit(X_train,y_train)


# In[120]:


rs.cv_results_


# In[121]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Parameters")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[122]:


#Gradient Boosting Machine after parameter tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(subsample = 0.7000000000000002,
        n_estimators = 62,
        min_samples_split = 350,
        min_samples_leaf = 190,
        max_depth = 3,
        learning_rate = 0.15)
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10, X_train, y_train)


# LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train,y_train)
pred_y = lgbm.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(lgbm,10, X_train, y_train)


# In[123]:


text_df.columns


# In[ ]:




