import pandas as pd

def order_simplify_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.drop_duplicates(subset=['date'], inplace=True)

    return df
