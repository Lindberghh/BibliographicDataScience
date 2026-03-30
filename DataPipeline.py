import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# given dataset, preprocesses all data for classification model, saves the preprocessor
df = pd.read_csv('data/converted_data/b3_new')
numerical_features = [
    "Number of Pages",
    "Date"
]
text_features = [
    "gnd Topical Term",
    "Title Statement",
    "Publisher",
    "Place",
    "Author"
]

german_stop_words = stopwords.words('german')
subset = df.groupby('gnd Genre', group_keys=False).sample(frac=1.0, random_state=67)

X = subset[["Publisher", "Title Statement", "Author", "gnd Topical Term", "Date", "Number of Pages"]].fillna("")
y = subset['gnd Genre']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=67,
    stratify=y
)
# preprocessing timeline title, topic, publisher -> TFIDF + SVD
preprocessor = ColumnTransformer(
    transformers=[
        (
             "title_svd",
             Pipeline([
                 ("tfidf", TfidfVectorizer(
                     max_features=40000,
                     stop_words=german_stop_words,
                     ngram_range=(1,2),
                     min_df=10,
                     max_df=0.9
                 )),
                 ("svd", TruncatedSVD(n_components=400, random_state=67))
             ]),
             "Title Statement"
         ),
         (
             "topic_svd",
             Pipeline([
                 ("tfidf", TfidfVectorizer(max_features=10000, min_df=10, max_df=0.9)),
                 ("svd", TruncatedSVD(n_components=100, random_state=67))
             ]),
             "gnd Topical Term"
          ),
        (
            "publisher_svd",
            Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10000, min_df=10, max_df=0.9)),
                ("svd", TruncatedSVD(n_components=100, random_state=67))
            ]),
            "Publisher"
        ),
        (
            "num",
            SimpleImputer(strategy="median", add_indicator=True),
            numerical_features
        )
    ],
    remainder="drop",
    n_jobs=-1
)
preprocessor.fit(X_train)
joblib.dump(preprocessor, "data/preprocessor_new_no_title_no_topic.joblib")