from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
stopwords = stopwords.words('english')


def main():
    project_path = str(Path(__file__).resolve().parents[1])
    data = pd.read_csv(project_path + '/data/imdb_reviews.csv')
    data = data.astype('object')
    print(data.head())

    processed_reviews = Parallel(n_jobs=8)(delayed(process_review)(row['review'])
                                           for i, row in tqdm(data.iterrows(), total=data.shape[0]))
    print(processed_reviews[0])

    data['review'] = processed_reviews
    train = data.sample(frac=0.6)
    freq_list = Counter()
    for review in train['review']:
        freq_list.update(review)
    print(len(freq_list))
    print(freq_list.most_common(5000)[-1])
    freq_list = freq_list.most_common(5000)

    # Reserve 0 for padding
    freq_list = {freq[0]: i + 1 for i, freq in enumerate(freq_list)}
    processed_reviews = Parallel(n_jobs=2)(delayed(filter_words)(review, freq_list) for review in tqdm(processed_reviews))

    # Split into train (60), validation (20), and test sets (20)
    data['review'] = processed_reviews
    train = data.iloc[train.index]
    data = data.drop(train.index)
    validation = data.sample(frac=0.5)
    test = data.drop(validation.index)

    train.to_csv(project_path + '/data/train.csv')
    validation.to_csv(project_path + '/data/validation.csv')
    test.to_csv(project_path + '/data/test.csv')


def process_review(review):
    # Remove html tags
    review = BeautifulSoup(review, "html.parser").get_text()
    review = review.lower()
    review = nltk.word_tokenize(review)

    # Remove stopwords/punctuation and stem words
    review = [stemmer.stem(w) for w in review if w not in stopwords]
    review = [w for w in review if contains_alphabet(w)]
    return review


def filter_words(review, freq_list):
    return [freq_list[word] for word in review if word in freq_list]


def contains_alphabet(word):
    return any([c.isalpha() for c in word])


if __name__ == "__main__":
    main()
