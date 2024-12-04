import pandas as pd
import numpy as np
from scipy.stats import rankdata
from datetime import timedelta
import os

# Define the directory containing the .npy files
directory = "Indicator_Matrices"

# List all .npy files in the directory
npy_files = [f for f in os.listdir(directory) if f.endswith("_indicator_matrix.npy")]

# Extract ticker names by removing the suffix
tickers = ['TRMB', 'HBI', 'M', 'ARKQ', 'LYV', 'PRU', 'BBY']
#print(tickers)

rolling_window = 100
investment_length = 4
quantile_var = 8
num_stocks = len(tickers)

def import_and_structure(list, data_period):

    def order_simplify_data(df):
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.drop_duplicates(subset=['date'], inplace=True)

        return df

    min_sup_date = None

    ticker_dict = {}
    for ticker in list:
        ticker_data = pd.read_csv(f'Ticker_Data/total_universe_data/{ticker}_1hr_historical_data_final.csv')
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        ticker_data_ordered = order_simplify_data(ticker_data)
        ticker_data_ordered['Hr'] = (ticker_data_ordered.groupby(ticker_data_ordered['date'].dt.date).cumcount() + 1)
        ticker_dict[ticker] = ticker_data_ordered

        if min_sup_date is None:
            min_sup_date = ticker_data_ordered['date'].max()
        elif min_sup_date >= ticker_data_ordered['date'].max():
            min_sup_date = ticker_data_ordered['date'].max()

    common_dates = None
    for ticker, ticker_data in ticker_dict.items():
        if common_dates is None:
            common_dates = set(ticker_data['date'])
        else:
            common_dates &= set(ticker_data['date'])

    ticker_arrays_dict = {}
    for ticker, ticker_data in ticker_dict.items():
        # Apply the `common_dates` filter and truncate to the specified period and common dates
        ticker_data_truncated = ticker_data[(ticker_data['date'].isin(common_dates)) &
                                            (ticker_data['date'] >= (min_sup_date-timedelta(weeks=data_period)))]

        ticker_array = ticker_data_truncated.loc[:,
                       ['date', 'Hr', 'open', 'high', 'low', 'close', 'volume']].to_numpy()
        ticker_arrays_dict[ticker] = ticker_array

    return ticker_arrays_dict

# index: date = 0, hr = 1, open = 2, high = 3, low = 4, close = 5, volume = 6

ticker_data_dictionary = import_and_structure(tickers, 50)


#print(ticker_data_dictionary['ARKQ'][:10, :3])

indicator_matrices = {}
for ticker in tickers:
    ticker_matrix = np.load(f'Indicator_Matrices/{ticker}_indicator_matrix.npy', allow_pickle=True)
    #print(ticker_matrix.shape)
    indicator_matrices[ticker] = ticker_matrix

#common dates of indicator matrices and ticker_data
common_dates = None
for ticker, indicator_data in indicator_matrices.items():
    if common_dates is None:
        common_dates = set(indicator_data[:, 0])
    else:
        common_dates &= set(indicator_data[:, 0])

#print(common_dates)
#print(indicator_matrices['ARKQ'].shape)

def calc_FR_and_stack(dictionary, investment_length, common_dates):
    FR_matrices = {}

    for ticker, ticker_data in dictionary.items():
        #ensure only using rows that have available quantiled indicator values for each stock


        #ensure starts on an open hour, finishes on a close hour, means there will be matching open/close hour pairs
        #index 0 = datetime, index 1 = hr, all on dim[1]
        if ticker_data[-1, 1] != 7:
            indices = np.where(ticker_data[:, 1] == 7)[0]
            last_index = indices[-1]
            ticker_data = ticker_data[:last_index+1, :]

        if ticker_data[0, 1] != 1:
            indices = np.where(ticker_data[:, 1] == 1)
            first_index = indices[0]
            ticker_data = ticker_data[first_index:, :]

        #filter for only open and close hour, check for days missing an open on close hour and remove its corresponding pair
        hr_column = ticker_data[:, 1].astype(int)
#        print(hr_column)
        filter_mask_both = (hr_column == 1) | (hr_column == 7)
        filtered_ticker_data_both = ticker_data[filter_mask_both]

        rows_to_keep = np.ones(filtered_ticker_data_both.shape[0], dtype=bool)
        opens_to_remove = []

        for i in range(1, filtered_ticker_data_both.shape[0]):
            if (filtered_ticker_data_both[i, 1] == 1) and (filtered_ticker_data_both[i - 1, 1] == 1):
                rows_to_keep[i-1] = False  # Remove consecutive open values
            elif (filtered_ticker_data_both[i - 1, 1] == 7) and (filtered_ticker_data_both[i, 1] == 7):
                rows_to_keep[i] = False  # Remove consecutive close values

        filtered_matrix = filtered_ticker_data_both[rows_to_keep, :]

        hr_column_filtered = filtered_matrix[:, 1].astype(int)
        #print(hr_column_filtered)

        #split finalised filtered matrix into open values and close values
        filter_mask_open = (hr_column_filtered == 1)
        filtered_ticker_data_open = filtered_matrix[filter_mask_open]
        #print(f'filtered_ticker_data_open: {filtered_ticker_data_open}')

        filter_mask_close = (hr_column_filtered == 7)
        filtered_ticker_data_close = filtered_matrix[filter_mask_close]

        #error checking
        if filtered_ticker_data_close.shape != filtered_ticker_data_open.shape:
            print('Error in number of open/close hour pairs')
            print(f'filtered_ticker_data_close shape: {filtered_ticker_data_close.shape}')
            print(f'filtered_ticker_data_open shape: {filtered_ticker_data_open.shape}')

        #calc FR's for every trading day with a closing hour pair x days in the future, where x is investment_length (no. of days invested in)
        FR_array = np.full(filtered_ticker_data_close.shape[0], np.nan)
        #leave last FR entries as Nan as there is not sufficient data to calc FR
        for time in range(filtered_ticker_data_close.shape[0]-investment_length):
            FR = (filtered_ticker_data_close[time+investment_length, 2] - filtered_ticker_data_open[time, 5]) / filtered_ticker_data_open[time, 5]
            FR_array[time] = FR

        FR_array = FR_array[:, np.newaxis]  # Add a new axis to make it 2D
        matrix_with_FR = np.hstack((filtered_ticker_data_open, FR_array))

        #create dictionary of all matrices to then later stack
        FR_matrices[ticker] = matrix_with_FR

    #stack stock matrices into one large matrix with all stock values, as well as forward return
    common_dates = common_dates
    for matrix in FR_matrices.values():
        common_dates &= set(matrix[:, 0])

    total_ticker_data = None
    for matrix in FR_matrices.values():
        matrix = matrix[np.isin(matrix[:, 0], list(common_dates))]
        if total_ticker_data is None:
            # Add the first matrix as the initial 3D array
            total_ticker_data = matrix[np.newaxis, :, :]  # Shape: (1, 210, 8)
        else:
            # Concatenate along the first axis
            total_ticker_data = np.concatenate((total_ticker_data, matrix[np.newaxis, :, :]), axis=0)

    #return opens_to_remove in order to filter out rows in indicator matrix that were filtered out above, to ensure proper alignment when comparing
    return total_ticker_data, common_dates

total_ticker_data, common_dates = calc_FR_and_stack(ticker_data_dictionary, investment_length, common_dates)
print('Finished calculating total_ticker_data')
print(f'total_ticker_data shape: {total_ticker_data.shape}')

def stack_rank_quantile(dictionary, quantile_var, common_dates):

    total_indicator_matrix = None
    for indicator_matrix in dictionary.values():
        indicator_matrix = indicator_matrix[np.isin(indicator_matrix[:, 0], list(common_dates))]
        if total_indicator_matrix is None:
            total_indicator_matrix = indicator_matrix[np.newaxis, :, :]
            #print(f'total_indicator_matrix shape: {total_indicator_matrix.shape}')
        else:
            total_indicator_matrix = np.concatenate((total_indicator_matrix, indicator_matrix[np.newaxis, :, :]), axis=0)
            #print(f'total_indicator_matrix shape: {total_indicator_matrix.shape}')

    def cross_rank_inds(matrix):
        ranked_indicator_matrix = np.full((matrix.shape[0], matrix.shape[1], matrix.shape[2]-1), np.nan)
        #datetime_array = np.empty((matrix.shape[0], matrix.shape[1]), dtype='datetime64[ns]')
        for time in range(matrix.shape[1]):
            #datetime_values = pd.Series(matrix[:, time, 0])
            #datetime_array[:, time] = datetime_values.to_numpy()
            for indicator in range(matrix.shape[2]-1):
                indicator_values = matrix[:, time, indicator+1]
                ranked_ind_values = (rankdata(indicator_values)-1)/(num_stocks-1)

                ranked_indicator_matrix[:, time, indicator] = ranked_ind_values
        return ranked_indicator_matrix #, datetime_array

    ranked_ind_mat = cross_rank_inds(total_indicator_matrix)

    def quantiler(matrix, quantile_var):
        quantiled_ind_matrix = np.full(matrix.shape, np.nan)
        for time in range(matrix.shape[1]):
            for indicator in range(matrix.shape[2]):
                data = matrix[:, time, indicator]

                quantile_edges = np.quantile(data, q=np.linspace(0, 1, quantile_var + 1))
                quantile_edges = np.unique(quantile_edges)

                quantile_labels = np.digitize(data, bins=quantile_edges[1:-1], right=True)+1
                quantile_labels = np.clip(quantile_labels, 1, quantile_var)

                quantiled_ind_matrix[:, time, indicator] = quantile_labels

        return quantiled_ind_matrix

    quantiled_ind_mat = quantiler(ranked_ind_mat, quantile_var)

    return quantiled_ind_mat #, datetime_array

total_quantiled_indicator_matrix= stack_rank_quantile(indicator_matrices, quantile_var, common_dates)
print('Finished calculating total_quantiled_indicator_matrix')

print('total_quantiled_indicator_matrix shape', total_quantiled_indicator_matrix.shape)
#print('datetime_array shape', datetime_array.shape)

np.save(f'Indicator_Matrices/test_total_feature_matrix.npy', total_quantiled_indicator_matrix)
np.save(f'Indicator_Matrices/test_total_ticker_data.npy', total_ticker_data)
