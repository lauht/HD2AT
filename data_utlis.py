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

def add_alpha(args, selected_tickers):
    print('Loading base technical data...')
    all_timestamps, all_tickers, all_data = get_data(
        start_time=args.prestart_time, 
        end_time=args.lagend_time, 
        selected_tickers=selected_tickers, 
        market='sp500',
        )
    data_with_alpha = all_data.copy()
    
    print('Loading indicators...')
    for comp in all_tickers:
        close_series = all_data.loc[:,'feature'].loc[:,'$close'].swaplevel().loc[comp, :]
        high_series = all_data.loc[:,'feature'].loc[:,'$high'].swaplevel().loc[comp, :]
        low_series = all_data.loc[:,'feature'].loc[:,'$low'].swaplevel().loc[comp, :]
        volume_series = all_data.loc[:,'feature'].loc[:,'$volume'].swaplevel().loc[comp, :]
        return_series = all_data.loc[:,'feature'].loc[:,'$close/Ref($close, 1)-1'].swaplevel().loc[comp, :]

        df_alpha = pd.DataFrame(close_series)
        types_all = ['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3',
                    'atr', 'natr', 'trange',
                    'rsi',

                    'obv', 'norm_return', 'obv-ref',
                    'macd', 'macdsignal', 'macdhist', 'macdhist-ref',
                    'slowk', 'slowd', 'norm_kdjhist', 'kdjhist-ref',]
        
        # MA
        types_ma=['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3']
        for i in range(len(types_ma)):
            df_alpha[types_ma[i]] = ZscoreNorm(ta.MA(close_series, timeperiod=5, matype=i))
        # Volatility Indicators
        df_alpha['atr'] = ZscoreNorm(ta.ATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['natr'] = ZscoreNorm(ta.NATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['trange'] = ZscoreNorm(ta.TRANGE(high_series, low_series, close_series))
        # RSI
        df_alpha['rsi'] = ta.RSI(close_series, timeperiod=5) / 100


        # Volume Indicators
        df_alpha['obv'] = ZscoreNorm(ta.OBV(close_series, volume_series))
        df_alpha['norm_return'] = ZscoreNorm(return_series)
        df_alpha['obv-ref'] = df_alpha['obv'] - df_alpha['obv'].shift(1)
        # MACD
        macd, macdsignal, macdhist = ta.MACD(close_series, fastperiod=12, slowperiod=26, signalperiod=9)
        df_alpha['macd'], df_alpha['macdsignal'], df_alpha['macdhist'] = ZscoreNorm(macd), ZscoreNorm(macdsignal), ZscoreNorm(macdhist)
        df_alpha['macdhist-ref'] = ZscoreNorm(df_alpha['macdhist'] - df_alpha['macdhist'].shift(1))
        # KDJ
        slowk, slowd = ta.STOCH(high_series, low_series, close_series, fastk_period=5, slowk_period=3)
        df_alpha['slowk'] = ZscoreNorm(slowk)
        df_alpha['slowd'] = ZscoreNorm(slowd)
        df_alpha['norm_kdjhist'] = ZscoreNorm(slowk - slowd)
        df_alpha['kdjhist-ref'] = ZscoreNorm((slowk-slowd) - (slowk-slowd).shift(1))


        newindex = pd.MultiIndex.from_product([df_alpha.index.to_list(), [comp]], names=['datetime', 'instrument'])
        df_alpha.set_index(newindex, inplace=True)
        for type in types_all:
            data_with_alpha.loc[(slice(None), comp), ('alpha', type)] = df_alpha[type]
            
    ## retrieve data from starttime to endtime
    data_with_alpha = data_with_alpha.loc[datetime.datetime.strptime(args.start_time, '%Y-%m-%d'):datetime.datetime.strptime(args.end_time, '%Y-%m-%d')]
    if pd.isnull(data_with_alpha.values).any():
        print('Exist Nan')

    final_timestamps = list(D.calendar(start_time=args.start_time, end_time=args.end_time))
    if len(data_with_alpha) == len(final_timestamps) * len(all_tickers):
        return final_timestamps, all_tickers, data_with_alpha
    else:
        raise Exception('Data is not aligned.')
    
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