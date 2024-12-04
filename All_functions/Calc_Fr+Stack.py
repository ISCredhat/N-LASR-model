import numpy as np

rolling_window = 100

indicator_matrices = {}
for ticker in tickers:
    ticker_matrix = np.load(f'Indicator_Matrices/{ticker}_indicator_matrix.npy', allow_pickle=True)
    #print(ticker_matrix.shape)
    indicator_matrices[ticker] = ticker_matrix

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







