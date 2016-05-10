# domainadaption

### 0. Available data
Amazon product reviews in four categories: books, dvd, electronics, and kitchen & housewares.<br>
1000 positive, 1000 negative and various unlabeled reviews per category.<br>
Data is available [here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html).
```
├── sorted_data_acl/
    ├── books/
    │   ├── negative.review
    │   ├── positive.review
    │   ├── unlabeled.review
    ├── dvd/
    │   ├── negative.review
    │   ├── positive.review
    │   ├── unlabeled.review
    ├── electronics/
    │   ├── negative.review
    │   ├── positive.review
    │   ├── unlabeled.review
    |── kitchen_&_houswares/
        ├── negative.review
        ├── positive.review
        ├── unlabeled.review
```

### 1. Create Embeddings
##### 1.1. Run *preprocess_dataset_for_embeddings.py*<br>
  This will create a *reviews_forEmbedding.txt* file in each category folder. The file will contain all reviews (positive, negative and unlabeled) of that categories with one sentence of a review per line. The sentences do not contain any special characters or any punctuation. 

##### 1.2. Run *sorted_data_acl/merge_reviews.sh*<br>
  This will merge all the above files into one file and store them in the sorted_data_acl/all/ folder.

##### 1.3. Run *create_word_embeddings.sh*<br>
  This will create word embeddings for each category (including *all*) of the reviews using [GloVe](https://github.com/stanfordnlp/GloVe). In particular, this creates the following 4 files in each category folder:
  * *reviews.vocab*: word count per category in the format word -> count
  * *reviews.cooccur*: cooccurance matrix of words
  * *reviews.cooccur.shuf*: sorted cooccurence matrix
  * *reviews.vectors.txt*: word embeddings per category in the format word -> vector
  
##### 1.4. Run *build_embedding_dictionary.py*<br>
  This will create Python dictionaries in the format word -> vector from the files *reviews.vectors.txt* and store it in the files *reviews.vectors.pkl*.

### 2. Transform Text Reviews into Embedded Reviews
##### 2.1. Run *preprocess_dataset.py*<br>
  This will create a *reviews_positive.txt*, *ratings_positive.txt*, *reviews_negative.txt* and *ratings_revative.txt* files in each category folder. The files will contain the respective reviews and ratings with one review/rating per line. 
##### 2.2. Run *embed_reviews.py*<br>
  This will transfrom the text reviews into embedded reviews by converting each word into a vector using the dictionaries from previous steps. The resulting matrices will be stores in *reviews_positive.npy* and *reviews_negative.npy*.
