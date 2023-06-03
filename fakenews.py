import numpy as np                                          #To create numpy arrays
import pandas as pd                                         #To create dataframes and storing data in the dataframes
import re                                                   #Regular Expression library is used to search words in texts & paragraphs
from nltk.corpus import stopwords                           #Natural Language Toolkit contains stopwords like articles/qns words
from nltk.stem.porter import PorterStemmer                  #Natural Language Toolkit contains stemwords that gives root words
from sklearn.feature_extraction.text import TfidfVectorizer #To convert text/parameters to feature vectors/numerical data
from sklearn.model_selection import train_test_split        #Splits data into training data & test data 
from sklearn.linear_model import LogisticRegression         #For binary classification, i.e, real(0) & fake(1) 
from sklearn.metrics import accuracy_score                  #Returns real/fake news accuracy
import nltk
nltk.download('stopwords')
#To print the stopwords in the English languages
print(stopwords.words('english'))
#To load the dataset to a pandas Dataframe
news_dataset=pd.read_csv('/Users/sahoo/Documents/train.csv')
#To print 1st 5 rows from the dataset
news_dataset.head()
#To print total articles in the dataset
news_dataset.shape
#To count number of missing values in the dataset
news_dataset.isnull().sum()
#To replace null values with empty string
news_dataset=news_dataset.fillna('')
#To merge author name and news title
news_dataset['content']=news_dataset['author']+': '+news_dataset['title']
print(news_dataset['content'])
#To separate data & label
X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']
print(X)
print(Y)
port_stem=PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english' or 'ENGLISH')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content
news_dataset['content']=news_dataset['content'].apply(stemming)
print(news_dataset['content'])
#To separate data & label
X=news_dataset['content'].values
Y=news_dataset['label'].values
print(X)
print(Y)
Y.shape
X.shape
#To convert textual data into numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
print(X)
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#Training the model
model=LogisticRegression()
model.fit(X_train, Y_train)
#Accuracy score of the model on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data=", training_data_accuracy)
#Accuracy score of the model on test data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the testing data=", testing_data_accuracy)
#Predicting the model
X_new=X_test[0]
prediction=model.predict(X_new)
print(prediction)
if(prediction==0):
    print("The news is real")
else:
    print("The news is fake")
print(Y_test[0])