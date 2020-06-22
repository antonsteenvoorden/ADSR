# -*- coding: utf-8 -*-
"""
@author: Anton Steenvoorden
"""
import pandas as pd
import csv
import numpy as np
import random
import argparse


def apply_filter(account_X):
    # drop duplicate interactions within the same session
    account_X.drop_duplicates(subset=['account', 'item_id', 'time'], keep='first', inplace=True)

    # keep items with >=5 interactions
    item_pop = account_X.item_id.value_counts()
    good_items = item_pop[item_pop >= MINIMUM_ITEM_INTERACTIONS].index
    account_X = account_X[account_X.item_id.isin(good_items)]

    # remove accounts with items < 5
    account_pop = account_X.account.value_counts()
    good_account = account_pop[account_pop >= MINIMUM_USER_INTERACTIONS].index
    account_X = account_X[account_X.account.isin(good_account)]

    # remove account with unique items < 5
    item_per_account = account_X.groupby('account')['item_id'].nunique()
    good_account_item = item_per_account[item_per_account >= MINIMUM_UNIQUE_ITEMS].index
    account_X = account_X[account_X.account.isin(good_account_item)]

    return account_X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='data/ML1M.data')

    MINIMUM_UNIQUE_ITEMS = 5
    MINIMUM_ITEM_INTERACTIONS = 20
    MINIMUM_USER_INTERACTIONS = 20
    args = parser.parse_args()

    data_origin = pd.read_csv(args.input_file, header=None, delimiter="\t",
                              names=['account', 'item_id', 'rate', 'time'], dtype={'rate': int})

    u_max_num = data_origin.account.nunique()
    v_max_num = data_origin.item_id.nunique()

    print('total user:%d' % u_max_num)
    print('total item:%d' % v_max_num)

    data = apply_filter(data_origin)
    data = data.sort_values(by=['account', 'time'])
    print('After filtering:')
    u_max_num = data.account.nunique()
    v_max_num = data.item_id.nunique()
    print('total user:%d' % u_max_num)
    print('total item:%d' % v_max_num)
    output_file = args.input_file[:-len(".data")] + "_account.csv"
    print("writing to", output_file)
    data.to_csv(output_file, header=['account', 'item_id', 'rate', 'time'], index=False)
