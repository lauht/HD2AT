import os
import sys
import math
import datetime
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm

import qlib
from qlib.data import D
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.loader import QlibDataLoader
from causalml.inference.meta import XGBTRegressor
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor

provider_uri = "F:/qlib/qlib_data/us_data" # data dir
qlib.init(provider_uri=provider_uri, region=REG_US)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class HiddenWarnings:
    def __enter__(self):
        warnings.filterwarnings("ignore")
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.resetwarnings()

def get_base_company(args):
    instruments = D.instruments(market='sp500')
    company_pr = D.list_instruments(instruments=instruments, start_time=args.start_time, end_time=args.end_time, as_list=True)
    company_pr.sort()
    company_pr.append('^gspc') # sp500
    return company_pr

def get_uplift_data(company_1, args):
    df = pd.read_csv('./dataset/uplift_data.csv')
    df = df.dropna()
    all_timestamps = list(D.calendar(start_time=args.start_time, end_time=args.end_time))
    for timestamp in all_timestamps:
        tickers = list(df.loc[timestamp].index)
        union = list(set(tickers) & set(company_1))
        company_1 = union
    company_1.sort(reverse=False)
    selected_data = df.loc[(slice(None), company_1), :]
    return company_1, selected_data

def get_interval(interval):
    day_start = datetime.datetime.strptime(interval[0], '%Y-%m-%d')
    day_end = datetime.datetime.strptime(interval[1], '%Y-%m-%d')
    return day_start, day_end

def get_batch_sum(com_ser, total_len, batch_size=5):
    sn = np.array(com_ser, dtype=np.double)
    com_ser_sum = []
    for com_seri in sn:
        com_seri_sum = []
        for i in range(math.ceil(total_len/batch_size)):
            begin = i * batch_size
            end = begin + batch_size
            batch = com_seri[begin:end]
            s = batch.sum()
            com_seri_sum.append(s)
        com_ser_sum.append(com_seri_sum)
    return np.array(com_ser_sum, dtype=np.double)

def get_corr(events, price_df, company_sel, method, events_num=1):
    partial_corr = np.zeros((len(company_sel), len(company_sel)))
    count = np.ones(len(company_sel)) * len(events)
    i = 0
    for events_name, interval in events.items():
        if i >= events_num:
            count = count - 1
            break
        day_start, day_end = method(interval)

        if day_start == None:
            count = count - 1
            continue
        if day_start > day_end:
            count = count - 1
            continue

        day_start = day_start - datetime.timedelta(days=1)
        price_timestamps = list(D.calendar(start_time=day_start, end_time=day_end))
        if len(price_timestamps) <= 2:
            count = count - 1
            continue
        df_slice = price_df.loc[day_start:day_end].loc[:,'feature'].loc[:,'return'].swaplevel().sort_index()
        com_ser = []
        for comp in company_sel: # other series
            if comp == ('^gspc' or '^dji' or '^ndx'):
                com_ser.append(df_slice.loc[comp, :].values)
            else:
                try:
                    if len(df_slice.loc[comp, :].values)==len(price_timestamps):
                        com_ser.append(df_slice.loc[comp, :].values)
                    else:
                        com_ser.append(np.zeros(len(price_timestamps)))
                        count[company_sel.index(comp)] -= 1
                except:
                    com_ser.append(np.zeros(len(price_timestamps))) # fill with 0
                    count[company_sel.index(comp)] -= 1

        ## commulative return
        if len(price_timestamps) < 30:
            sn = np.array(com_ser, dtype=np.double)
        else:
            sn = get_batch_sum(com_ser, len(price_timestamps), batch_size=4)
            
        corr = np.corrcoef(sn)
        corr[np.isnan(corr)] = 0

        for a in range(0, sn.shape[0]-1):
            for b in range(0, sn.shape[0]-1):
                r_ab = corr[a, b]
                r_ac = corr[a, -1]
                r_bc = corr[b, -1]
                r_ab_c = (r_ab-r_ac*r_bc)/(((1-r_ac**2)**0.5)*((1-r_bc**2)**0.5)) # partial correlation coefficient
                partial_corr[a, b] += r_ab_c
        i += 1
    for a in range(0, sn.shape[0]-1):
        for b in range(0, sn.shape[0]-1):
            partial_corr[a, b] /= min(count[a], count[b])
    return partial_corr

def gen_adj(dir_adj, lim_pos, lim_neg):
    cond_pos = dir_adj>=lim_pos
    cond_neg = dir_adj<=lim_neg
    dir_adj.where(cond_pos | cond_neg, other=0, inplace=True)
    dir_adj_np = np.array(dir_adj)
    return dir_adj_np
