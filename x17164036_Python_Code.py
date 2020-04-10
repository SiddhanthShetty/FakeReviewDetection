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
stop = stopwords.words('english')


# In[2]:


#Importing Raw Data
reviews_df = pd.read_csv('D:\\NCI Notes\\Thesis\\Data\\Reviews.csv', encoding = 'cp1252')
restaurant_df = pd.read_csv('D:\\NCI Notes\\Thesis\\Data\\Restaurant.csv', encoding = 'cp1252')


# In[3]:


#Checking for Missing values - Reviews Dataset
column_names = reviews_df.columns
print(column_names) #Column names
totalCells = np.product(reviews_df.shape) #Calculate total number of cells in dataframe
missingCount = reviews_df.isnull().sum() #Count number of missing values per column
totalMissing = missingCount.sum() #Calculate total number of missing values 
print("The Reviews dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.") #Calculate percentage of missing values


# In[4]:


#Replace any missing values with NA and Drop NA values
reviews_df.replace(' ',np.nan, inplace = True)
reviews_df = reviews_df.dropna()


# In[5]:


#Deleting rows flagged as NR and YR
reviews_df = reviews_df[reviews_df.flagged != 'NR']
reviews_df = reviews_df[reviews_df.flagged != 'YR']


# In[6]:


#Encoding categorical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
reviews_df['flagged_code'] = le.fit_transform(reviews_df['flagged'])


# In[7]:


#Unique Id column
reviews_df['ReviewID'] = reviews_df.index

#Adding review content wordcount feature
reviews_df['WordCount_Review'] = reviews_df['reviewContent'].apply(lambda comment: len(comment.split()))

#Drop unwanted columns
reviews_df = reviews_df.drop(columns=['date','reviewID','coolCount','funnyCount'], axis = 1)

#Renaming Columns
reviews_df.rename(columns={'rating':'ReviewRating','usefulCount':'UsefulCount_Review'}, inplace = True)


# In[8]:


#Renaming Columns - Restaurant dataset
restaurant_df.rename(columns={'reviewCount':'TotalReviewCountofRestaurant','filReviewCount':'FakeReviewCountRestaurant','rating':'AggRestaurantRating'}, inplace = True)
rest_df = restaurant_df[['restaurantID','TotalReviewCountofRestaurant','FakeReviewCountRestaurant','AggRestaurantRating']]


# In[9]:


#Joining the two datasets
combined_df = pd.merge(reviews_df,rest_df, how = 'inner', on = 'restaurantID')
combined_df = pd.DataFrame(combined_df)
combined_df.to_csv(r'D:\\NCI Notes\\Thesis\\Data\\FinalMergedDataset.csv')


# In[10]:


combined_df.head()


# In[11]:


combined_df.info()


# In[12]:


combined_df.describe()


# In[13]:


#Text preprocessing and new feature addition
#Remove numbers
combined_df['reviewContentNew'] = combined_df['reviewContent'].apply(lambda x: ''.join([i for i in x if not i.isdigit()])) 
#Remove punctuations
combined_df['reviewContentNew'] = combined_df['reviewContentNew'].str.replace('[^\w\s]', '')
#Lowercasing
combined_df['reviewContentNew'] = combined_df['reviewContentNew'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#stopword count
combined_df['stopwordsCount'] = combined_df['reviewContentNew'].apply(lambda x: len([x for x in x.split() if x in stop]))
#removing stopwords
combined_df['reviewContentNew'] = combined_df['reviewContentNew'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#lemmatizing
combined_df['reviewContentNew'] = combined_df['reviewContentNew'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[14]:


combined_df['reviewContentNew'].head()


# In[15]:


#Adding new numeric features - Feature Engineering
#Rating features and Text based statistics
#AggRating Deviation
combined_df['DeviationfromAggRating'] = abs(combined_df['AggRestaurantRating'] - combined_df['ReviewRating'])
#Character count
combined_df['charCount'] = combined_df['reviewContent'].apply(len)
#uppercase count
combined_df['uppercaseCount'] = combined_df['reviewContent'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
#special char count
combined_df['specialCharCount'] = combined_df['reviewContent'].apply(lambda comment: sum (comment.count(w) for w in '[\w]+'))
#sentence count
combined_df['sentenceCount'] = combined_df['reviewContent'].apply(ts.sentence_count)


# In[16]:


#Redability features
combined_df['fleschReadingEase'] = combined_df['reviewContent'].apply(ts.flesch_reading_ease)
combined_df['fleschKincaidGrade'] = combined_df['reviewContent'].apply(ts.flesch_kincaid_grade)
combined_df['fogScale'] = combined_df['reviewContent'].apply(ts.gunning_fog)
combined_df['smogScore'] = combined_df['reviewContent'].apply(ts.smog_index)
combined_df['ARI'] = combined_df['reviewContent'].apply(ts.automated_readability_index)
combined_df['CLI'] = combined_df['reviewContent'].apply(ts.coleman_liau_index)
combined_df['linsearWrite'] = combined_df['reviewContent'].apply(ts.linsear_write_formula)
combined_df['daleChallScore'] = combined_df['reviewContent'].apply(ts.dale_chall_readability_score)


# In[17]:


#Sentiment Score
#combined_df['sentimentScore'] = combined_df['reviewContent'].apply(lambda x: TextBlob(x).sentiment[0])#SentimentScore-polarity


# In[18]:


#POS Tagging and adding its counts as features
from nltk import word_tokenize, pos_tag
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

combined_df['nounCount'] = combined_df['reviewContent'].apply(count_noun)
combined_df['verbCount'] = combined_df['reviewContent'].apply(count_verb)
combined_df['adjectiveCount'] = combined_df['reviewContent'].apply(count_adjective)
combined_df['adverbCount'] = combined_df['reviewContent'].apply(count_adverb)


# In[19]:


combined_df.info()


# In[20]:


#Exporting the PreProcessed Data
combined_df.to_csv(r'D:\\NCI Notes\\Thesis\\Data\\CombinedPreProcessedDataset.csv')


# In[21]:


#Defining functions for Report, Cross Validation

def model_report(y_act, y_pred):
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
    print("Mean accuracy: ", result.mean())
    print("Variance: ", result.std())
    pass

def KFold_Cross_Validation(classifier,n):
    from sklearn.model_selection import cross_val_score
    acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = n, scoring = 'accuracy')
    cross_validation_report(acc)
    pass


# In[22]:


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
text_df.to_csv(r'D:\\NCI Notes\\Thesis\\Data\\UndersampledDataset.csv')


# In[29]:


#Handling Class Imbalance
#Up-sampling using SMOTE for only numeric features model(Rating Features and Text Based Features)
numeric_df = combined_df.drop(columns=['reviewerID','reviewContent','flagged','ReviewID','restaurantID','reviewContentNew'])
X = numeric_df.drop(columns=['flagged_code','fleschReadingEase',
       'fleschKincaidGrade', 'fogScale', 'smogScore', 'ARI', 'CLI',
       'linsearWrite', 'daleChallScore'])
X = X.iloc[:,:].values
y = numeric_df.iloc[:,numeric_df.columns.get_loc('flagged_code')].values


# In[30]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_sample(X, y)


# In[31]:


#Class instances after upsampling using SMOTE
print("Classes after SMOTE : ")
print("Count of label '1' = {}".format(sum(y==1)))
print("Count of label '0' = {}".format(sum(y==0)))


# In[32]:


#Performing feature selection on numeric features using BorutaPy
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
feat_selector = BorutaPy(rfc, n_estimators='auto',verbose=2, random_state=0)
feat_selector.fit(X,y)


# In[33]:


#Updating feature set for training with selected features
X = X[:,feat_selector.support_]


# In[34]:


#Split Dataset for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 123)


# In[35]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred_y = gnb.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gnb,10)


# In[36]:


#XGBoost without parameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10)


# In[37]:


#Hyperparameter Tuning for XGBoost
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


# In[38]:


rs.cv_results_


# In[39]:


print("Best accuracy Obtained: {0}".format(rs.best_score_))
print("Best Parameters: ")
for key, value in rs.best_params_.items():
    print("\t{}:{}".format(key, value))


# In[40]:


#XGBoost after Parameter tuning
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
KFold_Cross_Validation(xgbc,10)


# In[41]:


#AdaBoost without Parameter tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10)


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
KFold_Cross_Validation(abc,10)


# In[46]:


#Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gnb,10)


# In[59]:


#XGBoost before Hyperparameter Tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10)


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
KFold_Cross_Validation(xgbc,10)


# In[64]:


#AdaBoost without parameter Tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10)


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
KFold_Cross_Validation(abc,10)


# In[69]:


#Gradient Boosting Machine without parameter tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gnb,10)


# In[84]:


#XGBoost before HyperParameter tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10)


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
KFold_Cross_Validation(xgbc,10)


# In[89]:


#AdaBoost before parameter tuning
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10)


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
KFold_Cross_Validation(abc,10)


# In[94]:


#Gradient Boosting Machine before tuning
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gnb,10)


# In[108]:


#XGBoost before HyperParameter tuning
import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train,y_train)
pred_y = xgbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(xgbc,10)


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
KFold_Cross_Validation(xgbc,10)


# In[113]:


#AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
pred_y = abc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(abc,10)


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
KFold_Cross_Validation(abc,10)


# In[118]:


#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
pred_y = gbc.predict(X_test)
model_report(y_test,pred_y)
KFold_Cross_Validation(gbc,10)


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
KFold_Cross_Validation(gbc,10)


# In[123]:


text_df.columns


# In[ ]:




