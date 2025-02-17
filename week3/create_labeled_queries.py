import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
from nltk.tokenize import word_tokenize

stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
max_depth = 0
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    # Determine the longest branch in the tree
    if len(cat_path_ids) > max_depth:
        max_depth = len(cat_path_ids)
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

print("Normalizing...")
# Convert queries to lower case, replace non-alphanumeric chars with a space and replace consecutive spaces with a single space
queries_df['query'] = queries_df['query'].str.lower().replace("[^0-9a-z]+", " ", regex=True).replace("\s+", " ", regex=True)

print("Stemming...")
# For every query, tokenize the query, stem each individual token, then rejoin and update the panda
for i, row in queries_df.iterrows():
    query = queries_df.at[i, 'query']
    tokens = word_tokenize(query)
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    stemmed_tokens = ' '.join(stemmed_tokens)
    queries_df.at[i,'query'] = stemmed_tokens

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

print("Rolling up categories...")
keep_looping = True
i = 1
# We can't roll up more times than the max depth
while keep_looping and i < max_depth:
    keep_looping = False
    for category, count in queries_df['category'].value_counts().items():
        if count < min_queries:
            # We found at least one category that has fewer than the threshold queries so we'll need to continue looking for more
            keep_looping = True
            parent_category = parents_df.loc[parents_df['category'] == category, 'parent'].item()
            queries_df.loc[queries_df['category'] == category, 'category'] = parent_category
    i+=1

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
