import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import tokenize
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# Import all methods from Preprocess.py here
# import *, imports all methods/ specify method name
from Preprocess import *

#nltk.download('vader_lexicon')

# Call preprocessing methods from proprocess.py here
# Calling a method from another file without defining class, so object is not required-- example
processText()

# Read the csv file into pandas dataframe
data = pd.read_csv("test.csv",encoding = "ISO-8859-1")

#Replace book id with the actual name present in html so as to iterate through html files
data['book_id'] = data['book_id'].str.replace('.epub', '-content.html', case = False)
book_id_array = data['book_id'].to_numpy()
#print(book_id_array[0])

genre = ['guten_genre']

#Convert each genre of data dataframe into numbers
for x in genre:
    le = LabelEncoder()
    le.fit(list(data[x].values))
    data[x] = le.transform(list(data[x]))
print('******************Dataframe after converting Genre as numbers**********************')
print(data)

wordArray = []
#Could be used for Doc2Vec
contentList = []
index = 0
tokensList = []
sentenceList = []

#Iterate through each .html files and read the contents and append to our exisitng dataframe
#while index < len(book_id_array):
while index < len(book_id_array):
	#Open the HTML file abd read the contents for tokenizing
	HTMLfile = open('E:\\OVGU\\Sem2_Summer2020\\Subjects\\ATML\\Semester_Project\\starter\\Gutenberg_English_Fiction_1k\\Gutenberg_19th_century_English_Fiction\\'+book_id_array[index],'r',encoding='utf-8')
	content = HTMLfile.read()
	r = re.compile('<.*?>')
	content_string = re.sub(r, '', content)
	#print('************************content_string****************************')
	#print(type(content_string))

	#Use this later for finding the length of sentence, Nothing to do with this for now
	sent_tokens = tokenize.sent_tokenize(content_string)
	print('*********************Sentence Tokens*************************************')
	#print(sent_tokens)
	print('Book: ',book_id_array[index],'Sentence Length: ',len(sent_tokens))

	prunc_removed_str = re.sub('[^A-Za-z0-9]+', ' ',content_string)
	print('Document ',index, 'is parsing..')
	#Tokenize words
	#tokens = word_tokenize(prunc_removed_str)

	#Append the words to form corpus
	#tokensList.extend(tokens)

	contentList.insert(index,prunc_removed_str)
	sentenceList.insert(index,len(sent_tokens))

	index = index+1
#After removing punctuations, storing the content in dataframe
data['content'] = contentList
#Pass the contents to SentimentIntensityAnalyser
sid = SentimentIntensityAnalyzer()
data['scores'] = data['content'].apply(lambda content: sid.polarity_scores(content))
# Compound is the value which determines if the doc is positive or negative ranges between -1 to +1, anything above 0 is positive
data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound'])
#Sentiment based on compound score
data['sentiment'] = data['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

print(data)

scoresList = list(data['scores'])
print('*************************Scores List*****************************')
#Consists of List of dict with the score having pos neg and neutral
print(scoresList)

#Seperate pos, neg and neutral and merge the array for vectorizing
posArray =[dic['pos'] for dic in scoresList]
negArray =[dic['neg'] for dic in scoresList]
neuArray =[dic['neu'] for dic in scoresList]
featureVec = np.vstack((posArray,neuArray, negArray)).T
print('Feature Vector',featureVec)
dfFeature = pd.DataFrame(data=featureVec, columns=['postive', 'neutral','negative'])
dfFeature['sentence_length'] = sentenceList

print('**************************Feature Df*****************************')
print(dfFeature)

#Split Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(dfFeature, data['guten_genre'], test_size=0.3)
print('*******************Train and Test Set Size**************************')
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


clf = MultinomialNB(alpha=.45)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('*******************F1 Score and Accuracy*****************************')
print('F1 Score: ',metrics.f1_score(y_test, pred, average='macro'))
print('Accuracy: ',metrics.accuracy_score(y_test, pred))






