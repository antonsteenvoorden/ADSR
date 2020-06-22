import argparse
import pandas as pd
import torch

HEADERS = ["Unknown", "Action", "Adventure", "Animation",
           "Children\'s", "Comedy", "Crime", "Documentary",
           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
           "Mystery", "Romance", "Sci-Fi",
           "Thriller", "War", "Western"]

genres = {key:i for i, key in enumerate(HEADERS)}

def binary_encoding(row):
    tmp_genres = []
    for g in row.split("|"):
        tmp_genres.append(genres[g])
    encoding = torch.zeros(len(HEADERS), dtype=torch.long)
    tmp_genres = torch.tensor(tmp_genres, dtype=torch.long)
    encoding[tmp_genres] = 1
    encoding = encoding.tolist()
    return encoding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_properties', default='movies_ml1m.dat')
    parser.add_argument('--output_file', default="ml1m.item")
    args = parser.parse_args()
    # 1::Toy Story (1995)::Animation|Children's|Comedy
    tmp_header = ["id", "name", "genres"]
    properties_df = pd.read_csv(args.input_properties, delimiter="::", header=None, names=tmp_header, encoding="ISO-8859-1")
    properties_df["genres"] = properties_df["genres"].apply(lambda x: binary_encoding(x))

    to_write = []
    for (id, g) in zip(properties_df["id"].values, properties_df["genres"].values):
        row = [id] + g
        to_write.append(row)
    tmp_df = pd.DataFrame(to_write)
    tmp_df.to_csv(args.output_file, index=False, sep="|", header=None)

