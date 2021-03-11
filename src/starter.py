import pandas as pd
# import seaborn as sns
import re
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

from texthero import stop_words
from sklearn.model_selection import train_test_split


def top_characters(proper_nouns, top_num):
    counts = dict(Counter(proper_nouns).most_common(top_num))
    return counts


def get_proper_nouns(text):
    proper_nouns = []
    i = 0
    proper_noun_set = set()
    while i < len(text):
        if text[i][1] == 'NNP':
            if text[i + 1][1] == 'NNP':
                proper_nouns.append(text[i][0].lower() + " " + text[i + 1][0].lower())
                i = i + 1
            else:
                proper_nouns.append(text[i][0].lower())
        i = i + 1
        proper_noun_set = set(proper_nouns)
    return proper_noun_set


def tagging(tokens):
    text = pos_tag(tokens)
    return text


def number_of_characters(_token):
    tagged_text = tagging(_token)
    proper_nouns = get_proper_nouns(tagged_text)
    print("Before stopwords",len(dict(Counter(proper_nouns))))
    proper_nouns = [w for w in proper_nouns if not w in stop_words]
    no_characters = len(dict(Counter(proper_nouns)))
    return no_characters


# Read the csv file into pandas dataframe
# data = pd.read_csv("/Users/nandish21/Downloads/1-Masters/2nd-Sem/ATML/Project/git/git_version_4/ATiML-Project-2020/ProjectCode/master996.csv", encoding="ISO-8859-1")
data = pd.read_csv("/Users/nandish21/Downloads/1-Masters/2nd-Sem/ATML/Project/git/git_version_4/ATiML-Project-2020/ProjectCode/master.csv", sep=',', encoding="ISO-8859-1")
# y_true = data['guten_genre'].values
# x_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify = y_true, test_size = 0.2)
# train_df, cv_df, y_train, y_cv = train_test_split(x_train, y_train, stratify = y_train, test_size = 0.2)
#
# print("Number of samples in training data :", train_df.shape[0])
# print("Number of samples in validation data :", cv_df.shape[0])
# print("Number of samples in test data :", test_df.shape[0])
#
# train_class_distribution = train_df['guten_genre'].value_counts().sort_index()
# cv_class_distribution = cv_df['guten_genre'].value_counts().sort_index()
# test_class_distribution = test_df['guten_genre'].value_counts().sort_index()
#
#
# print("\ntraining distribution:\n\n",train_class_distribution)
# print("\ncv distribution: \n\n", cv_class_distribution)
# print("\ntest distribution: \n\n", test_class_distribution)


# Replace book id with the actual name present in html so as to iterate through html files
data['book_id'] = data['book_id'].str.replace('.epub', '-content.html', case=False)
book_id_array = data['book_id'].to_numpy()
# print(book_id_array[0])

genre = ['guten_genre']

# Convert each genre of data dataframe into numbers
for x in genre:
    le = LabelEncoder()
    le.fit(list(data[x].values))
    data[x] = le.transform(list(data[x]))
print('******************Dataframe after converting Genre as numbers**********************')
print(data)
data['character_number'] = 0

wordArray = []
bigramArray = []
index = 0
count = 0

# Iterate through each .html files and read the contents and append to our exisitng dataframe
while index < len(book_id_array):
    # Open the HTML file abd read the contents for tokenizing
    # Replace the path with the folder where html files are present
    HTMLfile = open(
        '/Users/nandish21/Downloads/1-Masters/2nd-Sem/ATML/Project/Gutenberg_English_Fiction_1k'
        '/Gutenberg_19th_century_English_Fiction/' +
        book_id_array[index], 'r', encoding='utf-8')
    content = HTMLfile.read()
    r = re.compile('<.*?>')
    content_removed_tags = re.sub(r, '', content)

    # Pick pre-defineds Stop words from nltk lib
    # stop_words = list(stopwords.words('english'))

    # Tokenize the text
    tokens = word_tokenize(content_removed_tags)
    num_characters = number_of_characters(content_removed_tags)
    # Extract character feature'
    data.at[index, 'character_number'] = num_characters
    # Convert into lower case
    tokens = [w.lower() for w in tokens]
    # print('Document ',index,':')
    # print('After Tokenizing: ',len(tokens))

    # Remove non alphabetic words/Removing punctuations
    words = [word for word in tokens if word.isalpha()]
    # print('After Removing Punctuations: ',len(words))

    # #Removing Stopwords
    # words = [w for w in words if not w in stop_words]
    # print('After Removing Stopwords: ',len(words))

    bigram = Counter(list(ngrams(words, 2)))
    wordArray.insert(index, words)
    bigramArray.insert(index, bigram)
    index = index + 1
print(len(wordArray))

data['tokens'] = wordArray
data['bigram'] = bigramArray
# Convert the list into strings, column should have tokens seperated by space
data['list_string'] = data['tokens'].apply(lambda x: ' '.join(map(str, x)))

print('******************Dataframe after appending Tokens**********************')


# Dataframe after appending the tokens of respective books into data['list_string']
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


    # tagged_text = tokens
    # proper_nouns = get_proper_nouns(tagged_text)
    # num_characters = dict(Counter(proper_nouns))
    # return num_characters


data1 = gender_roles(data)
# print(data1.bigram[9])
print(data['character_number'])

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
