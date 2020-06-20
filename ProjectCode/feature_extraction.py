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
from lexical_diversity import lex_div as ld
from nltk.util import ngrams
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
import codecs
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from collections import OrderedDict
# import nlp
import spacy
from textstat.textstat import textstatistics, legacy_round
import en_core_web_sm

nlp = en_core_web_sm.load()


# nltk.download('vader_lexicon')

# Call preprocessing methods from proprocess.py here
# Calling a method from another file without defining class, so object is not required-- example
# processText()
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


def sentence_break(sentence):
    # nlp = spacy.load('en')
    doc = nlp(sentence)
    return doc.sents


def word_count(sentence):
    sentences = sentence_break(sentence)
    words = 0
    for sen in sentences:
        words += len([token for token in sen])
    return words


def sentence_count(sentence):
    sentences = sentence_break(sentence)
    count = 0
    for sen in sentences:
        count += 1
    return count


def avg_sentence_length(sentence):
    words = word_count(sentence)
    sentences = sentence_count(sentence)
    if sentences == 0:
        return 0
    else:
        average_sentence_length = float(words / sentences)
        return average_sentence_length


def syllables_count(word):
    return textstatistics().syllable_count(word)


def avg_syllables_per_word(sentence):
    syllable = syllables_count(sentence)
    words = word_count(sentence)
    if words == 0:
        return 0
    else:
        average_syllables_per_word = float(syllable) / float(words)
        return legacy_round(average_syllables_per_word, 1)


def flesch_reading_score(sentence):
    """
        Implementing Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        where
          ASL is average sentence length (number of words
                divided by number of sentences)
          and ASW is average word length in syllables (number of syllables
                divided by number of words)
    """
    flesch_read_score = 206.835 - float(1.015 * avg_sentence_length(sentence)) - float(
        84.6 * avg_syllables_per_word(sentence))
    return legacy_round(flesch_read_score, 2)


if __name__ == '__main__':
    # Read the csv file into pandas dataframe
    data = pd.read_csv("master996.csv", encoding="ISO-8859-1")

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

    wordArray = []
    # Could be used for Doc2Vec
    contentList = []
    index = 0
    tokensList = []
    sentenceList = []
    ttrList = []
    topicList = []
    genderHe = []
    genderShe = []
    bigramArray = []
    personCountList = []

    # Iterate through each .html files and read the contents and append to our exisitng dataframe
    # while index < len(book_id_array):
    while index < len(book_id_array):
        # Open the HTML file abd read the contents for tokenizing
        HTMLfile = open(
            '/Users/nandish21/Downloads/1-Masters/2nd-Sem/ATML/Project/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/' +
            book_id_array[index], 'r', encoding='utf-8')
        content = HTMLfile.read()
        r = re.compile('<.*?>')
        content_string = re.sub(r, '', content)

        # Use this later for finding the length of sentence, Nothing to do with this for now
        sent_tokens = tokenize.sent_tokenize(content_string)
        print('*********************Sentence Tokens*************************************')
        # print(sent_tokens)
        print('Book: ', book_id_array[index], 'Sentence Length: ', len(sent_tokens))

        prunc_removed_str = re.sub('[^A-Za-z0-9]+', ' ', content_string)
        print('Document ', index, 'is parsing..')
        # Tokenize words
        tokens = ld.tokenize(prunc_removed_str)
        ttr = ld.hdd(tokens)
        ttrList.insert(index, ttr)
        # Append the words to form corpus
        # tokensList.extend(tokens)

        tokens = word_tokenize(content_string)
        labels = dict([(str(x), x.label_) for x in nlp(str(content_string)).ents])

        person_count = [name for name, ent in labels.items() if ent == 'PERSON']
        words = [word for word in tokens if word.isalpha()]

        filtered_words = [w for w in words if not w in stopwords.words('english')]

        bigram_topic = list(ngrams(filtered_words, 2))
        bigram = Counter(list(ngrams(words, 2)))
        id2word = gensim.corpora.Dictionary(bigram_topic)
        wordArray.insert(index, words)
        bigramArray.insert(index, bigram)
        # id2word.filter_extremes(no_below=10, no_above=0.35)
        # id2word.compactify()
        # corpus = [id2word.doc2bow(text) for text in bigram_topic]
        # lda_model_tfidf = gensim.models.LdaMulticore(corpus, num_topics=5, id2word=id2word)
        # top_topics = lda_model_tfidf.get_document_topics(corpus, minimum_probability=0.0)
        # topic_vec = [top_topics[i][1][1] for i in range(5)]
        for i in topic_vec:
            topicList.insert(index, i)

        contentList.insert(index, prunc_removed_str)
        sentenceList.insert(index, len(sent_tokens))
        personCountList.insert(index, len(person_count))

        # bigram = Counter(list(ngrams(words, 2)))
        # wordArray.insert(index, words)
        # bigramArray.insert(index, bigram)

        index = index + 1
    # After removing punctuations, storing the content in dataframe

    data['tokens'] = wordArray
    data['bigram'] = bigramArray
    data['content'] = contentList
    # Pass the contents to SentimentIntensityAnalyser
    sid = SentimentIntensityAnalyzer()
    data['scores'] = data['content'].apply(lambda content: sid.polarity_scores(content))
    # Compound is the value which determines if the doc is positive or negative ranges between -1 to +1, anything above 0 is positive
    data['compound'] = data['scores'].apply(lambda score_dict: score_dict['compound'])
    # Sentiment based on compound score
    data['sentiment'] = data['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')
    flesch_reading_score_list = []

    gender_roles(data)

    flesch_reading_score_list = []
    for idx, sentence in enumerate(data['content']):
        flesch_read_score = flesch_reading_score(sentence)
        flesch_reading_score_list.insert(idx, flesch_read_score)
    data['Ease Of Readability'] = flesch_reading_score_list

    print(data)

    scoresList = list(data['scores'])
    print('*************************Scores List*****************************')
    # Consists of List of dict with the score having pos neg and neutral
    print(scoresList)

    # Seperate pos, neg and neutral and merge the array for vectorizing
    posArray = [dic['pos'] for dic in scoresList]
    negArray = [dic['neg'] for dic in scoresList]
    neuArray = [dic['neu'] for dic in scoresList]
    featureVec = np.vstack((posArray, neuArray, negArray)).T
    print('Feature Vector', featureVec)
    dfFeature = pd.DataFrame(data=featureVec, columns=['postive', 'neutral', 'negative'])
    dfFeature['sentence_length'] = sentenceList
    dfFeature['TTR'] = ttrList
    dfFeature['person_count'] = personCountList
    dfFeature['she_pronoun'] = data['She Pronoun']
    dfFeature['he_pronoun'] = data['He Pronoun']
    dfFeature['ease_of_readability'] = data['Ease Of Readability']
    dfFeature['class'] = data['guten_genre']

    dfFeature.to_csv(r'export_dataframe.csv', index=False, header=True)
    # dfFeature['topic'] = topicList
    print('**************************Feature Df*****************************')
    print(dfFeature)
