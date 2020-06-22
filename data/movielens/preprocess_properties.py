import argparse
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

"""
Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.
"""

HEADERS = ["Unknown", "Action", "Adventure", "Animation",
           "Children", "Comedy", "Crime", "Documentary",
           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
           "Mystery", "Romance", "Sci-Fi",
           "Thriller", "War", "Western"]


def remap_columns(data, c):
    """Remap the values in each to the fields in `columns` to the range [1, number of unique values]"""
    uniques = data[c].unique()
    col_map = pd.Series(index=uniques, data=np.arange(0, len(uniques)))
    str = c
    data[str] = col_map[data[c]].values
    return data, col_map


def get_new_categories(user_df):
    tmp_df = user_df.copy()
    tmp_df['category'] = tmp_df[user_df.columns[2:]].apply(lambda x: torch.tensor(x.values), axis=1)
    assignments = defaultdict(int)
    category_counts = torch.zeros(len(user_df.columns[2:]), dtype=torch.long)

    # first go over singles to get initial distribution
    for key, categories in zip(tmp_df['item_id'].values, tmp_df['category'].values):
        nonzeroes = categories.nonzero()
        if len(nonzeroes) > 1:
            continue

        if len(nonzeroes) == 1:
            cat = nonzeroes[0]
            assignments[key] = int(cat)
            category_counts[cat] += 1

    for key, categories in zip(tmp_df['item_id'].values, tmp_df['category'].values):
        # for key, categories in zip(tmp_df['item_id'].values[:100], tmp_df['category'].values[:100]):
        nonzeroes = categories.nonzero()
        if len(nonzeroes) <= 1:
            continue

        # if this movie is known, add the right 1 hot vector and continue
        if assignments[key] != 0:
            category_counts[assignments[key]] += 1
        else:
            nonzeroes = nonzeroes.squeeze()
            counts = category_counts[nonzeroes]
            assign_to = torch.argmin(counts).tolist()
            assign_to = nonzeroes[assign_to]
            category_counts[assign_to] += 1
            assignments[key] = int(assign_to)
    return assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/ML1M_account.csv')
    parser.add_argument('--input_properties', default='data/ml1m.item')
    parser.add_argument('--maps', default='data/ml1m_maps.csv')
    parser.add_argument('--output_file', default='data/ml1m_properties.csv')
    # parser.add_argument('--maps', default='data/ml1m_maps_10.csv')
    # parser.add_argument('--output_file', default='data/ml1m_properties_10.csv')
    args = parser.parse_args()

    if "ml1m" in args.input_properties:
        HEADERS = HEADERS[1:]

    mapping = pd.read_csv(args.maps, delimiter=",", header=0, usecols=["new_id", "item_id"],
                          dtype={"new_id": int, "item_id": int})
    # mapping from old to new
    mapping = {key: value for key, value in zip(mapping["item_id"].values, mapping["new_id"].values)}
    properties_df = pd.read_csv(args.input_properties, delimiter="|", header=None, encoding="ISO-8859-1")

    # Add headers with known column
    tmp_header = ["item_id"] + list(range(len(properties_df.columns) - 1))
    properties_df.columns = tmp_header

    # Only keep the items that exist after filtering, and map it to the new id
    properties_df = properties_df[properties_df["item_id"].isin(mapping.keys())]
    properties_df["item_id"] = properties_df["item_id"].map(mapping)  # now has new id

    # Sort according to new item_id
    properties_df = properties_df.sort_values(by=["item_id"])

    # Only use properties, so we can easily load them as tensors
    cols = [0] + list(range(-len(HEADERS), 0))
    properties_df = properties_df.iloc[:, cols]
    cols = ["item_id"] + HEADERS
    properties_df.columns = cols

    # First map the ids to the new ids
    user_df = pd.read_csv(args.input_file, delimiter=",", usecols=["account", "item_id"])
    user_df = user_df[user_df["item_id"].isin(mapping.keys())]
    user_df["item_id"] = user_df["item_id"].map(mapping)  # now has new ids
    user_df = user_df.merge(properties_df, on=["item_id"], how="left")  # merge the categories
    user_df.head()

    category_mapping = get_new_categories(user_df)  # item_id, category
    properties_df['category'] = properties_df['item_id'].map(category_mapping) + 1  # Padding
    padding_row = pd.DataFrame({h: 0 for h in properties_df.columns}, index=[0])
    properties_df = pd.concat([padding_row, properties_df], sort=False)

    # re order again so that we could use the last columns as categories
    cols = ["item_id", "category"] + HEADERS
    properties_df = properties_df[cols]

    properties_df.to_csv(args.output_file, index=False)
