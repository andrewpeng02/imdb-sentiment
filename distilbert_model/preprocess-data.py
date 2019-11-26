from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer

from nltk.corpus import stopwords

stopwords = stopwords.words('english')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def main():
    project_path = str(Path(__file__).resolve().parents[1])
    data = pd.read_csv(project_path + '/data/imdb_reviews.csv')
    data = data.astype('object')
    print(data.head())

    processed_reviews = Parallel(n_jobs=1)(delayed(process_review)(row['review'], distilbert_tokenizer)
                                           for i, row in tqdm(data.iterrows(), total=data.shape[0]))
    data['review'] = processed_reviews
    print(processed_reviews[0])

    # Split into train (60), validation (20), and test sets (20)
    train = data.sample(frac=0.6)
    data = data.drop(train.index)
    validation = data.sample(frac=0.5)
    test = data.drop(validation.index)

    train.to_csv(project_path + '/data/train.csv')
    validation.to_csv(project_path + '/data/validation.csv')
    test.to_csv(project_path + '/data/test.csv')


def process_review(review, distilbert_tokenizer=None):
    # Remove html tags
    review = BeautifulSoup(review, "html.parser").get_text()
    review = review.lower()
    review = distilbert_tokenizer.tokenize(review)

    # Remove stopwords/punctuation
    review = [w for w in review if w not in stopwords and contains_alphabet(w)]
    return review


def contains_alphabet(word):
    return any([c.isalpha() for c in word])


if __name__ == "__main__":
    main()
