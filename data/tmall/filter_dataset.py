import pandas as pd
import time
import argparse

MINIMUM_UNIQUE_ITEMS = 5
MINIMUM_LENGTH = 20
MINIMUM_ITEM_INTERACTIONS = 20
MINIMUM_CATEGORY_INTERACTIONS = 20

USER_KEY = 'use_ID'
ITEM_KEY = 'ite_ID'
CATEGORY_KEY = 'cat_ID'
TIME_KEY = 'time'
ACTION_KEY = 'act_ID'

def apply_filter(account_X):
    # drop duplicate interactions within the same session
    account_X.drop_duplicates(subset=[USER_KEY, ITEM_KEY, TIME_KEY], keep='first', inplace=True)

    # keep items with >=X interactions
    item_pop = account_X[ITEM_KEY].value_counts()
    good_items = item_pop[item_pop >= MINIMUM_ITEM_INTERACTIONS].index
    account_X = account_X[account_X[ITEM_KEY].isin(good_items)]

    # keep categories with >=X interactions
    cat_pop = account_X[CATEGORY_KEY].value_counts()
    good_cats = cat_pop[cat_pop >= MINIMUM_CATEGORY_INTERACTIONS].index
    account_X = account_X[account_X[CATEGORY_KEY].isin(good_cats)]

    # remove accounts with items < 5
    account_pop = account_X[USER_KEY].value_counts()
    good_account = account_pop[account_pop >= MINIMUM_LENGTH].index
    account_X = account_X[account_X[USER_KEY].isin(good_account)]

    # remove account with unique items < 5
    item_per_account = account_X.groupby(USER_KEY)[ITEM_KEY].nunique()
    good_account_item = item_per_account[item_per_account >= MINIMUM_UNIQUE_ITEMS].index
    account_X = account_X[account_X[USER_KEY].isin(good_account_item)]

    return account_X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ijcai16 (ijcai2016_taobao.csv) https://tianchi.aliyun.com/dataset/dataDetail?dataId=53
    """
    User_id: unique user id
    Seller_id: unique online seller id
    Item_id: unique item id
    Category_id: unique category id
    Online_Action_id: “0” denotes “click” while “1” for “buy”
    Time_Stamp: date of the format “yyyymmdd”
    """

    parser.add_argument("--input_file", default='data/tmall.csv')
    args = parser.parse_args()

    data_origin = pd.read_csv(args.input_file, sep=',', header=0, usecols=[0, 2, 3, 4, 5],
                              names=[USER_KEY, ITEM_KEY, CATEGORY_KEY, ACTION_KEY, TIME_KEY])
    data_origin[TIME_KEY] = pd.to_datetime(data_origin[TIME_KEY], format="%Y%m%d")

    BUY_EVENT = 1
    VIEW_EVENT = 0
    data_origin = data_origin[data_origin[ACTION_KEY] == BUY_EVENT]

    # dropping duplicates
    # data_origin = data_origin.drop_duplicates()
    data_origin = data_origin.drop([ACTION_KEY], axis=1)
    u_max_num = data_origin[USER_KEY].nunique()
    v_max_num = data_origin[ITEM_KEY].nunique()
    c_max_num = data_origin[CATEGORY_KEY].nunique()
    print('total user:%d' % (u_max_num))
    print('total item:%d' % (v_max_num))
    print('total categories:%d' % c_max_num)

    data = apply_filter(data_origin)
    data = data.sort_values(by=[USER_KEY, TIME_KEY])

    print('\nAfter filtering:')
    u_max_num = data[USER_KEY].nunique()
    v_max_num = data[ITEM_KEY].nunique()
    c_max_num = data[CATEGORY_KEY].nunique()
    print('total user:%d' % u_max_num)
    print('total item:%d' % v_max_num)
    print('total categories:%d' % c_max_num)
    print('total interactions:%d' % data.size)

    output_file = args.input_file[:-len(".csv")] + "_buy.csv"

    print("writing to", output_file)
    data.to_csv(output_file, header=["user_id", "item_id", CATEGORY_KEY, TIME_KEY], index=False)
