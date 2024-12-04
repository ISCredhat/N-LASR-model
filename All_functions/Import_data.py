import pandas as pd
from datetime import timedelta

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
                                            (ticker_data['date'] >= (min_sup_date - timedelta(weeks=data_period)))]

        ticker_array = ticker_data_truncated.loc[:,
                       ['date', 'Hr', 'open', 'high', 'low', 'close', 'volume']].to_numpy()
        ticker_arrays_dict[ticker] = ticker_array

    return ticker_arrays_dict

# index: date = 0, hr = 1, open = 2, high = 3, low = 4, close = 5, volume = 6