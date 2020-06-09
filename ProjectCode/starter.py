import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import nltk
import os
import codecs
import re
import string
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from collections import OrderedDict

# Read the csv file into pandas dataframe
data = pd.read_csv("master.csv",encoding = "ISO-8859-1")

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
bigramArray = []
index = 0
count = 0

#Iterate through each .html files and read the contents and append to our exisitng dataframe
while index < len(book_id_array):
	#Open the HTML file abd read the contents for tokenizing
	#Replace the path with the folder where html files are present
	HTMLfile = open('G:\\Semester 3\\ATML\\Project\\Gutenberg_English_Fiction_1k\\Gutenberg_English_Fiction_1k\\Gutenberg_19th_century_English_Fiction\\'+book_id_array[index],'r',encoding='utf-8')
	content = HTMLfile.read()
	r = re.compile('<.*?>')
	content_removed_tags = re.sub(r, '', content)

	#Pick pre-defineds Stop words from nltk lib
	# stop_words = list(stopwords.words('english'))


	# Tokenize the text
	tokens = word_tokenize(content_removed_tags)
	# Convert into lower case
	tokens = [w.lower() for w in tokens]
	# print('Document ',index,':')
	# print('After Tokenizing: ',len(tokens))

	#Remove non alphabetic words/Removing punctuations
	words = [word for word in tokens if word.isalpha()]
	# print('After Removing Punctuations: ',len(words))

	# #Removing Stopwords
	# words = [w for w in words if not w in stop_words]
	# print('After Removing Stopwords: ',len(words))

	bigram = Counter(list(ngrams(words, 2)))
	wordArray.insert(index,words)
	bigramArray.insert(index, bigram)
	index = index+1
print(len(wordArray))

data['tokens'] = wordArray
data['bigram'] = bigramArray
#Convert the list into strings, column should have tokens seperated by space
data['list_string'] = data['tokens'].apply(lambda x: ' '.join(map(str, x)))

print('******************Dataframe after appending Tokens**********************')
#Dataframe after appending the tokens of respective books into data['list_string']
# print(data)

# Feature - Gender roles
def gender_roles(data):
	he_count = []
	she_count = []
	pronouns = ['he', 'she']
	for idx, book in enumerate(data.bigram):
		c1 = 0
		c2 = 0
		for key, val in list(book.items()):
			if key[0] not in pronouns:
				del book[key]
			if key[0] == pronouns[0]:
				c1 += 1

			elif key[0] == pronouns[1]:
				c2 += 1
		he_count.insert(idx, c1)
		she_count.insert(idx, c2)
	data['He Pronoun'] = he_count
	data['She Pronoun'] = she_count
	return data

data1 = gender_roles(data)
print(data1.bigram[9])



# vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
#                             analyzer='word', token_pattern=r'\w+', use_idf=True,
#                             smooth_idf=True, sublinear_tf=True, stop_words = 'english')
# vectors = vectorizer.fit_transform(data['list_string'])
# print('************************Vectors Shape*******************************')
# print(vectors.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(vectors, data['guten_genre'], test_size=0.3)
# print('*******************Train and Test Set Size**************************')
# print (X_train.shape)
# print (y_train.shape)
# print (X_test.shape)
# print (y_test.shape)
#
#
# clf = MultinomialNB(alpha=.45)
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
#
# print('*******************F1 Score and Accuracy*****************************')
# print('F1 Score: ',metrics.f1_score(y_test, pred, average='macro'))
# print('Accuracy: ',metrics.accuracy_score(y_test, pred))


