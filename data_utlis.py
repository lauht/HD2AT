import os
import torch
import datetime
import numpy as np
import pandas as pd

import talib as ta
import qlib
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.constant import REG_CN, REG_US

provider_uri = "F:/qlib/qlib_data/us_data" # data dir
qlib.init(provider_uri=provider_uri, region=REG_US)

def get_base_company(args):
    instruments = D.instruments(market='sp500')
    company_pr = D.list_instruments(instruments=instruments, start_time=args.start_time, end_time=args.end_time, as_list=True)
    company_pr.sort()
    company_pr.append('^gspc') # sp500
    return company_pr

def get_stocks_nonan(company_1, args):
    df = pd.read_csv('./dataset/return_data.csv')
    df = df.dropna()
    all_timestamps = list(D.calendar(start_time=args.start_time, end_time=args.end_time))
    for timestamp in all_timestamps:
        tickers = list(df.loc[timestamp].index)
        union = list(set(tickers) & set(company_1))
        company_1 = union
    company_1.sort(reverse=False)
    selected_data = df.loc[(slice(None), company_1), :]

    return company_1, selected_data

def get_common(company_0, adj_path):
    adj_listdir = os.listdir(adj_path)

    adj_matrix_list = []
    for file in adj_listdir:
        matrix = np.load(os.path.join(adj_path, file))
        for i in range(len(matrix)):
            matrix[i, i] = 1
        adj_matrix_list.append(matrix)

    company_list_list = []
    causal_df_list = []
    for matrix in adj_matrix_list:
        causal_df = pd.DataFrame(matrix.copy(), index=company_0, columns=company_0)
        company_1 = []
        for comp in company_0:
            if causal_df.loc[:, comp].sum()<=1 and causal_df.loc[comp, :].sum()<=1:
                causal_df = causal_df.drop(labels=comp, axis=1)
                causal_df = causal_df.drop(labels=comp, axis=0)
            else:
                company_1.append(comp)
        company_list_list.append(company_1)
        causal_df_list.append(causal_df)

    for i, company_list in enumerate(company_list_list):
        if i == 0:
            company_union = set(company_list)
        else:
            company_union = company_union & set(company_list)
    company_union = list(company_union)
    company_union.sort()

    causal_df_list_final = []
    for i, company_list in enumerate(company_list_list):
        causal_df = causal_df_list[i]
        for comp in company_list:
            if comp not in company_union:
                causal_df = causal_df.drop(labels=comp, axis=1)
                causal_df = causal_df.drop(labels=comp, axis=0)
        causal_df_list_final.append(causal_df)
        
    return company_union, causal_df_list_final

def get_common2(company_0, company_1, adj_path):
    adj_listdir = os.listdir(adj_path)

    adj_matrix_list = []
    for file in adj_listdir:
        matrix = np.load(os.path.join(adj_path, file))
        for i in range(len(matrix)):
            matrix[i, i] = 1
        adj_matrix_list.append(matrix)

    causal_df_list = []
    for matrix in adj_matrix_list:
        causal_df = pd.DataFrame(matrix.copy(), index=company_0, columns=company_0)
        causal_df_list.append(causal_df)

    causal_df_list_final = []
    for causal_df in causal_df_list:
        for comp in company_0:
            if comp not in company_1:
                causal_df = causal_df.drop(labels=comp, axis=1)
                causal_df = causal_df.drop(labels=comp, axis=0)
        causal_df_list_final.append(causal_df)
        
    return causal_df_list_final

def get_data(start_time, end_time, selected_tickers, market='sp500'):
    if selected_tickers is None:
        instruments = D.instruments(market=market)
        all_tickers = (D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True))
        ## get selected tickers and selected data
        selected_tickers = all_tickers # initialize tickers
    else:
        instruments = selected_tickers
    window1, window2, window3 = 5, 20, 60
    
    ## get timestamps
    all_timestamps = list(D.calendar(start_time=start_time, end_time=end_time))
    
    ## get data
    all_data = pd.read_csv('./dataset/all_data.csv')
    all_data = all_data.dropna()

    for timestamp in all_timestamps:
        tickers = list(all_data.loc[timestamp].index)
        union = list(set(tickers) & set(selected_tickers))
        selected_tickers = union

    selected_tickers.sort(reverse=False)
    selected_data = all_data.loc[(slice(None), selected_tickers), :]

    ## examine alignment
    if len(selected_data) == len(all_timestamps) * len(selected_tickers):
        return all_timestamps, selected_tickers, selected_data
    else:
        raise Exception('Data is not aligned.')
    
def ZscoreNorm(series):
    return (series-np.mean(series))/np.std(series)
    
def get_features_n_labels(args, selected_tickers):
    final_timestamps, all_tickers, data_with_alpha = add_alpha(args, selected_tickers=selected_tickers)

    num_times = len(final_timestamps)
    num_nodes = len(all_tickers)
    num_features_n_label = data_with_alpha.shape[1]

    raw_data = torch.Tensor(data_with_alpha.values)
    features_n_labels = raw_data.reshape(num_times, num_nodes, num_features_n_label) # time, nodes, feature
    features = features_n_labels[:, :, 1:]
    labels = features_n_labels[:, :, 0]
    return features, labels, all_tickers, final_timestamps

def get_adj(selected_tickers, company_final, causal_df):
    for comp in selected_tickers:
        if comp not in company_final:
            causal_df = causal_df.drop(labels=comp, axis=1)
            causal_df = causal_df.drop(labels=comp, axis=0)
    cond_pos = causal_df>0
    cond_neg = causal_df<0
    causal_df.where(cond_pos & cond_neg, other=0, inplace=True)
    causal_df[cond_pos] = 1
    causal_df[cond_neg] = -1

    causal_adj = causal_df.to_numpy()
    adj = torch.Tensor(causal_adj)
    return adj

def get_hetero_adj(causal_df_list, selected_tickers, company_final):
    adj_list = []
    for causal_df in causal_df_list:
        adj = get_adj(selected_tickers, company_final, causal_df)
        adj_list.append(adj)
    return torch.stack(adj_list, dim=0)
