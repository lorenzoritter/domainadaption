#!/bin/bash

CATEGORIES="books
dvd
electronics
kitchen_&_housewares"

SENTIMENTS="positive
negative"

DATAPATH="sorted_data_acl/"

VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=50
MAX_ITER=10
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=10

for CATEGORY in $CATEGORIES
do
    for SENTIMENT in $SENTIMENTS
    do
        REVIEWFILE=$DATAPATH$CATEGORY"/"$SENTIMENT"_reviews_forEmbedding.txt"
        VOCABFILE=$DATAPATH$CATEGORY"/"$SENTIMENT"_reviews.vocab"
        COOCCUREFILE=$DATAPATH$CATEGORY"/"$SENTIMENT"_reviews.cooccur"
        COOCCUREFILE_SHUF=$DATAPATH$CATEGORY"/"$SENTIMENT"_reviews.cooccur.shuf"
        SAVEFILE=$DATAPATH$CATEGORY"/"$SENTIMENT"_reviews.vectors"


        echo $REVIEWFILE
        glove/build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $REVIEWFILE > $VOCABFILE
        glove/build/cooccur -memory $MEMORY -vocab-file $VOCABFILE -verbose $VERBOSE < $REVIEWFILE > $COOCCUREFILE
        glove/build/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCUREFILE > $COOCCUREFILE_SHUF
        glove/build/glove -save-file $SAVEFILE -threads $NUM_THREADS -input-file $COOCCUREFILE_SHUF -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCABFILE -verbose $VERBOSE
    done
done
