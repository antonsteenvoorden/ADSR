# -*- coding: utf-8 -*-
"""
@author: Anton Steenvoorden
"""
import pandas as pd
import csv
import numpy as np
import pickle
from functools import reduce
import argparse

TEST_SPLIT = 0.8

USER_KEY = 'account'
ITEM_KEY = 'item_id'
TIME_KEY = 'time'


def remap_columns(data, c):
    """Remap the values in each to the fields in `columns` to the range [1, number of unique values]"""
    uniques = data[c].unique()
    col_map = pd.Series(index=uniques, data=np.arange(1, len(uniques) + 1))
    str = c
    data[str] = col_map[data[c]].values
    return data, col_map


#####-----------For DSR-----------###############
def last_percentage_out_split(data, split=0.8,
                              clean_test=True,
                              min_session_length=2):
    """
    last_percentage_out_split
    splits actions done by a user to the train and test set with a split based on the sequence length
    """
    train_indices = []
    test_indices = []

    for key, user_seq in data.groupby(USER_KEY):
        split_point = int(split * len(user_seq.index))
        train_indices.extend(user_seq[:split_point].index.tolist())
        tmp_test_indices = user_seq[split_point:].index.tolist()
        if len(tmp_test_indices) >= min_session_length:
            test_indices.extend(user_seq[split_point:].index.tolist())

    train = data.iloc[train_indices]
    test = data.iloc[test_indices]

    # Keep only the test sequences where all items occur in the training set
    if clean_test:
        train_items = train[ITEM_KEY].unique()
        print("Before filtering test length is", len(test))
        to_remove = test.loc[~test[ITEM_KEY].isin(train_items)]
        to_remove = to_remove[USER_KEY].unique()
        test = test[~test[USER_KEY].isin(to_remove)]
        print("After filtering items that occur in training, length is", len(test))

        #  remove user sequences in test shorter than min_session_length
        filtered_test_indices = []
        for key, user_seq in test.groupby(USER_KEY):
            tmp_test_indices = user_seq.index.tolist()
            if len(tmp_test_indices) >= min_session_length:
                filtered_test_indices.extend(user_seq.index.tolist())

        test = data.iloc[filtered_test_indices]
        print("After filtering short sequences, length is", len(test))

    return train, test



def get_user_sequences_from_df(data):
    dic_DSR = list(data.groupby(USER_KEY))
    sequence_DSR = []
    for i in range(len(dic_DSR)):
        sub_seq_DSR = []
        account = dic_DSR[i][0]
        item = list(dic_DSR[i][1][ITEM_KEY])
        time = list(dic_DSR[i][1][TIME_KEY])

        sub_seq_DSR += (account, item, time)
        sequence_DSR.append(sub_seq_DSR)

    return sequence_DSR


#####-----------For DSR-----------###############
def DSR(click_seq, max_len):
    user_sequences = []
    targets = []
    users = []

    for i in range(len(click_seq)):
        acc_seq = click_seq[i]
        user_id = acc_seq[0]
        item = acc_seq[1]
        tmp_max_len = max_len

        # if the item sequence is too short
        if len(item) <= max_len:
            tmp_max_len = len(item)

        for j in range(0, tmp_max_len-1):
            target = item[j + 1]
            targets.append([target])

            users.append([user_id])
            # user sequence with padding if needed
            user_sequence = np.array(item[0:j + 1], dtype=np.int32)
            user_sequence = np.pad(user_sequence, (0, max_len - 1 - len(user_sequence)), 'constant',
                                   constant_values=(0))
            user_sequence = list(user_sequence)
            user_sequences.append(user_sequence)
        # if longer we make more windows
        if len(item) > max_len:
            for j in range(1, len(item) - max_len + 1):
                target = item[j + max_len - 1]
                targets.append([target])

                users.append([user_id])
                user_sequences.append(item[j:j + max_len - 1])

    return user_sequences, targets, users


def split_validation(test_df):
    df_copy = test_df.copy()
    num_users = df_copy[USER_KEY].nunique()
    split = np.random.permutation(num_users)

    valid_users = split[:int(num_users / 2)]
    test_users = split[int(num_users / 2):]

    valid = df_copy[df_copy[USER_KEY].isin(valid_users)]
    test = df_copy[df_copy[USER_KEY].isin(test_users)]
    return valid, test


def write_to_file(filepath, data):
    with open(filepath, 'w') as file:
        csv_out = csv.writer(file, delimiter="|")
        csv_out.writerow(['sequence', 'target', 'user'])
        for row in zip(*data):
            csv_out.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/ML1M_account.csv", type=str)  # ML1M_account.csv
    parser.add_argument("--item_file", default="data/ml1m.item",  type=str)
    parser.add_argument("--output_train", default="data/ml1m_train",  type=str)
    parser.add_argument("--output_valid", default="data/ml1m_valid",  type=str)
    parser.add_argument("--output_test", default="data/ml1m_test",  type=str)
    parser.add_argument("--output_sequences", default="data/ml1m_sequences_full",  type=str)
    parser.add_argument("--maps", default="data/ml1m_maps", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_len", default=10, type=int)
    args = parser.parse_args()
    args.output_train = f"{args.output_train}_{args.max_len}.csv"
    args.output_valid = f"{args.output_valid}_{args.max_len}.csv"
    args.output_test = f"{args.output_test}_{args.max_len}.csv"
    args.output_sequences = f"{args.output_sequences}_{args.max_len}.csv"
    args.maps = f"{args.maps}_{args.max_len}.csv"

    MAX_LEN = args.max_len
    np.random.seed(args.seed)

    print(args)

    S_account = pd.read_csv(args.input_file, usecols=[USER_KEY, ITEM_KEY, TIME_KEY])
    S_account = S_account.sort_values(by=[USER_KEY, TIME_KEY])
    item_genre = pd.read_table(args.item_file, sep='|', header=None, encoding="ISO-8859-1")

    u_max_num = S_account.account.nunique()
    v_max_num = S_account.item_id.nunique()
    print('total user:%d' % (u_max_num))
    print('total item:%d' % (v_max_num))

    S_account, _ = remap_columns(S_account, USER_KEY)
    S_account, col_map = remap_columns(S_account, ITEM_KEY)
    col_map = pd.DataFrame({'new_id': col_map.values, ITEM_KEY: col_map.index})
    col_map = pd.merge(col_map, item_genre, left_on=ITEM_KEY, right_on=0).drop(0, axis=1)
    col_map.to_csv(args.maps, index=False)

    train_df, test_df = last_percentage_out_split(S_account, split=TEST_SPLIT)
    valid_df, test_df = split_validation(test_df)

    # store full sequences
    sequences_full = train_df.append(valid_df)
    sequences_full = sequences_full.append(test_df)
    # sequences_full = sequences_full.groupby(USER_KEY)[ITEM_KEY].apply(list)
    # sequences_full.to_csv(args.output_sequences, header=False, index=False)

    # Calculate sparsity
    # N_ITEMS = sequences_full[ITEM_KEY].max()
    # N_USERS = sequences_full[USER_KEY].nunique()
    # matrix = np.zeros((N_USERS, N_ITEMS))
    # for index, (group_name, group) in enumerate(sequences_full.groupby(USER_KEY)):
    #     active = np.zeros(N_ITEMS)
    #     for row_index, row in group.iterrows():
    #         item = row[ITEM_KEY]
    #         active[int(item)-1] = 1
    #     matrix[index, :] = active
    # sparsity = (matrix > 0).sum()/np.prod(matrix.shape)
    # print(1-sparsity) # 0,9492914821

    u_max_num = train_df.account.nunique()
    v_max_num = train_df.item_id.nunique()
    print('total user:%d' % (u_max_num))
    print('total item:%d' % (v_max_num))

    train_sequences = get_user_sequences_from_df(train_df)
    valid_sequences = get_user_sequences_from_df(valid_df)
    test_sequences = get_user_sequences_from_df(test_df)

    # tuple of sequence, user, item
    train_seq_windowed, train_targets, train_accounts = DSR(train_sequences, MAX_LEN)
    valid_seq_windowed, valid_targets, valid_accounts = DSR(valid_sequences, MAX_LEN)
    test_seq_windowed, test_targets, test_accounts = DSR(test_sequences, MAX_LEN)

    train = (train_seq_windowed, train_targets, train_accounts)
    valid = (valid_seq_windowed, valid_targets, valid_accounts)
    test = (test_seq_windowed, test_targets, test_accounts)

    print("Writing to file")
    write_to_file(args.output_train, train)
    write_to_file(args.output_valid, valid)
    write_to_file(args.output_test, test)

    print('Done')
    print('Length of train: %d, valid: %d, test: %d, total: %d' % (len(train_seq_windowed), len(valid_seq_windowed),
                                                                   len(test_seq_windowed),
                                                                   len(train_seq_windowed) + len(valid_seq_windowed)
                                                                   + len(test_seq_windowed)))
