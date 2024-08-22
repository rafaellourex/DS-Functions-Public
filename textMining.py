import pandas as pd
import numpy as np


def NLTK_classify_sentiment(input_string):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    # VADER sentiment analysis tool
    sia = SentimentIntensityAnalyzer()

    # Get polarity scores
    polarity = sia.polarity_scores(input_string)

    # Return sentiment:
    # - "positive" if compound score >= 0.05
    # - "negative" if compound score <= -0.05
    # - "neutral" otherwise
    if polarity["compound"] >= 0.05:
        return (polarity["compound"], "positive")
    elif polarity["compound"] <= -0.05:
        return (polarity["compound"], "negative")
    else:
        return (polarity["compound"], "neutral")


# calculates the frequency of each word in a corpus
def word_counter(text_list):
    """
    Function that receives a list of strings and returns the (absolute) frequency of each word in that list of strings.
    """
    words_in_df = " ".join(text_list).split()

    # Count all words
    freq = pd.Series(words_in_df).value_counts()
    return freq
    # counts how many words there are per observation


def count_words(df, column):
    df["word_count"] = df[f"{column}"].apply(lambda x: len(str(x).split(" ")))
    # df[['abstract','word_count']].head()
    return df


# this function identifies stopwords in english
def identify_stopwords(data, col):
    # Count all words
    all_words = " ".join(data[f"{col}"]).split()
    freq = pd.Series(all_words).value_counts()
    freq_ = len(freq)

    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet
    from nltk.stem import SnowballStemmer
    from bs4 import BeautifulSoup
    import string

    stop = set(stopwords.words("english"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    stop_w = []
    for i in freq[:freq_].index:

        if i in stop:

            stop_w.append(i)

    pct = len(stop_w) / freq_
    print(f"{pct}% of the most frequent words are stop words")


def get_top_n_grams(corpus, top_k, n):
    """
    Function that receives a list of documents (corpus) and extracts
        the top k most frequent n-grams for that corpus.

    :param corpus: list of texts
    :param top_k: int with the number of n-grams that we want to extract
    :param n: n gram type to be considered
             (if n=1 extracts unigrams, if n=2 extracts bigrams, ...)

    :return: Returns a sorted dataframe in which the first column
        contains the extracted ngrams and the second column contains
        the respective counts
    """
    vec = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = []
    for word, idx in vec.vocabulary_.items():
        words_freq.append((word, sum_words[0, idx]))

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k])
    top_df.columns = ["Ngram", "Freq"]
    return top_df


def plot_frequencies(top_df):
    """
    Function that receives a dataframe from the "get_top_n_grams" function
    and plots the frequencies in a bar plot.
    """
    x_labels = top_df["Ngram"][:30]
    y_pos = np.arange(len(x_labels))
    values = top_df["Freq"][:30]
    plt.bar(y_pos, values, align="center", alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel("Frequencies")
    plt.title("Words")
    plt.xticks(rotation=90)
    plt.show()


def remove_word(text, word):
    """
    Function that receives a list of strings and removes a specified token from every string in which it appears.
    """

    return re.sub(f"{word}", "", text)


# this function receives a dataframe and performs basic pre-processing for text mining,
# (lowercase, remove stopwords and punctuation performing lemmatization and stemming isbb optional)


def clean(text_list, lemmatize, stemmer, stopword):
    from tqdm import tqdm_notebook as tqdm

    import re
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet
    from nltk.stem import SnowballStemmer
    from bs4 import BeautifulSoup
    import string

    stop = set(stopwords.words("english"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    snowball_stemmer = SnowballStemmer("english")

    """
    Function that a receives a list of strings and preprocesses it.
    
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    updates = []
    for j in tqdm(range(len(text_list))):

        text = text_list[j]

        # LOWERCASE TEXT
        text = text.lower()

        # REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z!?]", " ", text)

        # REMOVE TAGS
        text = BeautifulSoup(text).get_text()

        if stopword == True:
            text = " ".join([word for word in text.split() if word not in (stop)])

        if lemmatize:
            text = " ".join(lemma.lemmatize(word) for word in text.split())

        if stemmer:
            text = " ".join(snowball_stemmer.stem(word) for word in text.split())

        updates.append(text)

    return updates


def clean_df(data, col, lemmatize=False, stemmer=False, stopword=False):
    from nltk.corpus import stopwords

    stop = set(stopwords.words("english"))
    updates = clean(
        data[f"{col}"], lemmatize=lemmatize, stemmer=stemmer, stopword=stopword
    )
    data["textClean"] = updates
    return data


def bag_words(train, dev, test, col, range1=1, range2=1):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    cv = CountVectorizer(
        max_df=0.8,
        binary=True,
        token_pattern='(?u)\\b\\w\\w+\\b|!|\?|"',
        ngram_range=(range1, range2),
    )
    train = cv.fit_transform(train[f"{col}"])
    dev = cv.transform(dev[f"{col}"])

    test = cv.transform(test[f"{col}"])
    return (train, dev, test, cv)


def extract_feature_scores(feature_names, document_vector):
    """
    Function that creates a dictionary with the TF-IDF score for each feature.
    :param feature_names: list with all the feature words.
    :param document_vector: vector containing the extracted features for a specific document

    :return: returns a sorted dictionary "feature":"score".
    """
    feature2score = {}
    for i in range(len(feature_names)):
        feature2score[feature_names[i]] = document_vector[0][i]
    return sorted(feature2score.items(), key=lambda kv: kv[1], reverse=True)


def TF_IDF_2(train, dev, test, cv):
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_vectorizer = TfidfTransformer()
    tfidf_vectorizer.fit(train)
    # get feature names
    feature_names = cv.get_feature_names()

    # fetch document for which keywords needs to be extracted

    # generate tf-idf for the given document
    tf_idf_train = tfidf_vectorizer.transform(train)
    tf_idf_dev = tfidf_vectorizer.transform(dev)
    tf_idf_test = tfidf_vectorizer.transform(test)

    scores = extract_feature_scores(feature_names, tf_idf_train.toarray())
    return (tf_idf_train, tf_idf_dev, tf_idf_test)
