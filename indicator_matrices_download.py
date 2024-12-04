import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import timedelta

tech_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSM', 'NVDA', 'ASML', 'ADBE', 'CSCO', 'ORCL',
           'INTC', 'CRM', 'TXN', 'QCOM', 'AVGO', 'IBM', 'AMD', 'INTU', 'NOW', 'SHOP', 'SNOW', 'SQ',
           'PYPL', 'UBER', 'LYFT', 'ZM', 'DOCU', 'TEAM', 'TWLO', 'NET', 'OKTA', 'CRWD', 'PANW', 'ZS',
           'DDOG','MDB', 'PLTR', 'FSLY', 'ESTC', 'PD', 'FROG', 'S', 'CIBR', 'BUG', 'VGT',
           'XLK', 'SOXX', 'SMH', 'XSD', 'FDN', 'ARKK', 'ARKW', 'ARKF',
           'ARKG', 'ARKQ', 'ARKX', 'ROBO', 'BOTZ', 'KOMP', 'WCLD', 'CLOU', 'SKYY',
           "ACN", "ADSK", "AKAM", "ANSS", "APH", "ASAN", "BB", "BIGC", "BILL", "BL",
           "BR", "BSY", "CDNS", "CDW", "CHKP", "CIEN", "CSGP", "CYBR", "DOCS",
           "DT", "EPAM", "ETSY", "FFIV", "FSLR", "FTNT", "GLW", "IAC", "KEYS", "KLAC",
           "LSCC", "MCHP", "MNDY", "MPWR", "MSI", "MU", "NTAP", "NXPI", "ON",
           "PATH", "PCTY", "RNG", "SEDG", "SSNC", "TER", "TRMB", "TTD", "UPST"]

healthcare_tickers = [
    "A", "ABBV", "ABT", "ACAD", "ACHC", "ADPT",
    "AGIO", "ALGN", "ALKS", "ALNY", "AMGN", "AMN",
    "ANAB", "ARWR", "ASND", "AXGN", "AXNX",
    "BAX", "BDX", "BIIB", "BIO", "BMRN", "BSX", "BMY", "CARA", "CCRN",
    "CERS", "CHRS", "CNC", "CNMD", "CORT", "COST", "COTY", "CRL",
    "CRSP", "CTLT", "CVS", "CYH", "DGX", "DHR", "DVA", "DXCM", "EBS",
    "EHC", "EHTH", "ELAN", "EOLS", "ESPR", "EXAS", "FGEN",
    "FMS", "FOLD", "GILD", "GMED", "GSK", "HCA",
    "HCSG", "HSIC", "HUM", "ICLR", "IDXX", "ILMN", "INCY",
    "INSP", "IONS", "IQV", "IRWD", "ISRG", "ITCI", "JAZZ", "JNJ", "KIDS",
    "LLY", "LMAT", "LNTH", "MCK", "MD", "MDT", "MOH", "MRK", "MYGN",
    "NTRA", "NVS", "OFIX", "OPK", "OSUR", "PFE", "PGNY", "PINC",
    "PRGO", "PRTA", "PTCT", "QDEL", "RGEN", "RMD", "RPRX", "SAGE", "SNY",
    "SRPT", "STE", "SYK", "TECH", "TMO", "UHS", "UNH", "UTHR", "VEEV", "VRTX",
    "VTRS", "WAT", "WST", "XRAY", "ZBH", "ZTS"]

financials_tickers = [
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "SCHW", "BK",
    "TFC", "COF", "AXP", "STT", "NTRS", "RF", "FITB", "KEY", "CFG",
    "HBAN", "MTB", "CMA", "ZION", "CFR", "PB", "BOKF", "CBSH", "WAL",
    "EWBC", "FHN", "SNV", "CUBI", "SF", "PJT", "EVR", "MC", "RJF",
    "HLI", "VIRT", "MKTX", "NDAQ", "CME", "ICE", "CBOE",
    "MSCI", "SPGI", "MCO", "FDS", "V", "MA", "PYPL", "SQ", "FIS", "GPN",
    "WU", "JKHY", "ADP", "PAYX", "WEX", "ENV", "BR", "SSNC", "FICO",
    "INTU", "CRM", "NOW", "WDAY", "HUBS", "TEAM", "DOCU", "TWLO", "OKTA",
    "ZS", "CRWD", "NET", "DDOG", "SNOW", "MDB", "ESTC", "PD", "FROG", "S",
    "CLOU", "CIBR", "BUG", "SKYY", "IGV", "VGT", "XLK", "SOXX", "SMH",
    "XSD", "FDN", "ARKK", "ARKW", "ARKF", "ARKG", "ARKQ",
    "ARKX", "ROBO", "BOTZ", "KOMP", "WCLD", "ALLY", "AMP", "AON", "APH",
    "ARES", "AFL", "AIG", "AIZ", "AJG",
    "AMG", "APO", "BEN", "BRO", "BX", "CINF", "CNA", "CNO",
    "CNS", "EQH", "FHI", "GL", "IVZ", "JHG", "KNSL", "L", "LNC", "LPLA",
    "MET", "MKL", "NAVI", "PGR", "PRI", "PRU", "RGA"]

energy_tickers = ['SLB', 'SO', 'SPH', 'SR', 'SRE', 'SUN', 'SWX', 'TRP', 'UGI', 'USAC',
           'EXC', 'FANG', 'FE', 'HAL', 'HE', 'HES', 'IDA', 'KMI', 'LNT', 'MPC',
           'MPLX', 'NEE', 'NI', 'NJR', 'NRG', 'NWE', 'NWN', 'OGE', 'OGS',
           'VLO', 'WEC', 'WES', 'WMB', 'XEL', 'XOM', 'EQT',
           'AEE', 'AEP', 'AES', 'ALE', 'AM', 'APA', 'AROC', 'ATO', 'AVA', 'BKR',
           'CMS', 'CNP', 'COP', 'CVI', 'CVX', 'D', 'DK', 'DTE', 'DUK', 'DVN',
           'ED', 'EE', 'EIX', 'ENB', 'ENLC', 'EOG', 'EPD', 'ES', 'ET', 'ETR',
           'OXY', 'PAA', 'PBF', 'PCG', 'PEG', 'PNW', 'POR', 'PPL', 'PSX',
           'MRO', 'RRC', 'AR', 'CRK', 'GPOR', 'MTDR', 'NOG', 'SM',
           "BEP", "CLNE", "CWEN", "DINO", "ENPH", "FLNG", "FTI", "GLNG", "HESM",
           "HNRG", "KOS", "MGY", "NFE", "NGL", "NVGS", "OBE", "OIS", "OR",
           "PAGP", "PARR", "PDS", "PLUG", "RUN", "TRGP", "VET", "VST",
           "YORW", "AES", "TPIC"]

consumer_tickers = [
    "AEO", "ULTA", "MAR", "WGO", "CWH", "DRI", "SNBR", "KSS", "WYNN",
    "APTV", "PHM", "TOL", "F", "CNK", "UAA", "ROST", "HBI", "DHI",
    "PTON", "M", "AMZN", "LEN", "EXPE", "SHOO", "NKE", "NVR", "POOL", "LKQ",
    "MHK", "BLMN", "LVS", "DG", "LULU", "DAR", "TPX", "RL", "AZO",
    "DLTR", "AN", "FOXA", "DENN", "BKNG", "CCL", "NWL", "W",
    "GNTX", "SBUX", "LYV", "CROX", "NWG", "MNRO", "JWN", "AAP",
    "WING", "MCD", "OMC", "LOW", "SJM", "TJX", "CBRL", "SKX", "TXRH",
    "BKE", "QSR", "BBWI", "REI", "TSLA", "HOG", "HRB", "SYY", "BBY",
    "FL", "JBLU", "YUM", "EBAY", "ORLY", "GRMN", "VC", "CZR",
    "RCL", "PLNT", "ALGT", "VFC", "GME", "TGT", "DIS",
    "WHR", "KMX", "TPR", "CAKE", "THO", "BWA", "GM", "PLAY", "LAD",
    "HD", "GPC", "LEG", "DPZ", "PVH", "HAS", "MELI", "DKS", "GES",
    "SPB", "ARCO", "URBN", "CMG", "PENN", "MGM", "EAT", "ETSY", "WSM", "ROKU"]

# Combine all lists and remove duplicates
#tickers = list(set(tech_tickers + healthcare_tickers + financials_tickers + energy_tickers + consumer_tickers))[:20]
tickers = ['TRMB', 'HBI', 'M']
#print(tickers)

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

def add_indicators(dictionary):
    indicator_matrices = {}

    rolling_window = 100

    # index: date = 0, hr = 1, open = 2, high = 3, low = 4, close = 5, volume = 6


    for ticker, ticker_matrix in dictionary.items():
        #name my matrix
        matrix = ticker_matrix
        indicator_matrix = None

        #iterate over each hour to check if first hour of day or not
        for time in range(rolling_window*2+1, matrix.shape[0]):

            #if it is first hour, calculate the indicator for the hour
            if matrix[time, 1] == 1:

                open_col = pd.Series(matrix[time-2*rolling_window:time, 2])
                high_col = pd.Series(matrix[time-2*rolling_window:time, 3])
                low_col = pd.Series(matrix[time-2*rolling_window:time, 4])
                close_col = pd.Series(matrix[time-2*rolling_window:time, 5])
                volume_col = pd.Series(matrix[time-2*rolling_window:time, 6])

                #normal indicators
                fisher = ta.fisher(high_col, low_col).iloc[:, 0].astype('float64')
                corr = close_col.rolling(window=rolling_window).corr(close_col.shift(1))
                adx = ta.adx(high_col, low_col, close_col)['ADX_14']
                rsi = ta.rsi(close_col)
                uo = ta.uo(high_col, low_col, close_col)
                willr = ta.willr(high_col, low_col, close_col)
                kc_upper = ta.kc(high_col, low_col, close_col).iloc[:, 0]
                kama = ta.kama(close_col)
                vtx = ta.vortex(high_col, low_col, close_col).iloc[:, 0]
                cmf = ta.cmf(high_col, low_col, close_col, volume_col)
                kst = ta.kst(close_col).iloc[:, 0]
                bop = ta.bop(open_col, high_col, low_col, close_col)
                cci = ta.cci(high_col, low_col, close_col)
                cmo = ta.cmo(close_col)
                percent_return = ta.percent_return(close_col)

                # non-normal indicators
                efi = (ta.efi(close_col, volume_col) - open_col) / open_col
                eri = (ta.eri(high_col, low_col, close_col).iloc[:, 0] - open_col) / open_col
                ema = (ta.ema(close_col) - open_col) / open_col
                hma = (ta.hma(close_col) - open_col) / open_col
                linreg = (ta.linreg(close_col) - open_col) / open_col
                slope = (ta.slope(close_col) - open_col) / open_col
                median = (ta.median(close_col) - open_col) / open_col
                macd = (ta.macd(close_col).iloc[:, 0] - open_col) / open_col
                mom = (ta.mom(close_col) - open_col) / open_col
                obv = (ta.obv(close_col, volume_col) - open_col) / open_col
                pvo = (ta.pvo(volume_col).iloc[:, 0] - open_col) / open_col
                roc = (ta.roc(close_col) - open_col) / open_col
                pvt = (ta.pvt(close_col, volume_col) - open_col) / open_col
                sma = (ta.sma(close_col) - open_col) / open_col
                stddev = (ta.stdev(close_col) - open_col) / open_col
                supertrend = (ta.supertrend(high_col, low_col, close_col).iloc[:, 0] - open_col) / open_col
                t3 = (ta.t3(close_col) - open_col) / open_col
                trima = (ta.trima(close_col) - open_col) / open_col
                true_range = (ta.true_range(high_col, low_col, close_col) - open_col) / open_col
                wma = (ta.wma(close_col) - open_col) / open_col
                tema = (ta.tema(close_col) - open_col) / open_col
                chop = (ta.chop(high_col, low_col, close_col) - open_col) / open_col
                atr = (ta.atr(high_col, low_col, close_col) - open_col) / open_col
                ao = (ta.ao(high_col, low_col) - open_col) / open_col
                dm = (ta.dm(high_col, low_col).iloc[:, 0] - open_col) / open_col
                dema = (ta.dema(close_col) - open_col) / open_col
                bias = (ta.bias(close_col) - open_col) / open_col
                cfo = (ta.cfo(close_col) - open_col) / open_col
                cti = (ta.cti(close_col) - open_col) / open_col
                inertia = (ta.inertia(close_col) - open_col) / open_col
                ppo = (ta.ppo(close_col).iloc[:, 0] - open_col) / open_col
                trix = (ta.trix(close_col).iloc[:, 0] - open_col) / open_col
                alma = (ta.alma(close_col) - open_col) / open_col
                fwma = (ta.fwma(close_col) - open_col) / open_col
                hilo = (ta.hilo(high_col, low_col, close_col).iloc[:, 0] - open_col) / open_col
                hl2 = (ta.hl2(high_col, low_col) - open_col) / open_col
                hwma = (ta.hwma(close_col) - open_col) / open_col
                midpoint = (ta.midpoint(close_col) - open_col) / open_col
                midprice = (ta.midprice(high_col, low_col) - open_col) / open_col
                pwma = (ta.pwma(close_col) - open_col) / open_col
                sinwma = (ta.sinwma(close_col) - open_col) / open_col
                kurtosis = (ta.kurtosis(close_col) - open_col) / open_col
                mad = (ta.mad(close_col) - open_col) / open_col
                skew = (ta.skew(close_col) - open_col) / open_col
                variance = (ta.variance(close_col) - open_col) / open_col

                # Create a dictionary mapping indicator names to their data series
                indicator_map = {
                    "fisher": fisher, "corr": corr, "adx": adx, "rsi": rsi, "uo": uo,
                    "willr": willr, "kc_upper": kc_upper, "kama": kama, "vtx": vtx, "cmf": cmf,
                    "kst": kst, "bop": bop, "cci": cci, "cmo": cmo, "percent_return": percent_return,
                    "efi": efi, "eri": eri, "ema": ema, "hma": hma, "linreg": linreg, "slope": slope,
                    "median": median, "macd": macd, "mom": mom, "obv": obv, "pvo": pvo, "roc": roc,
                    "pvt": pvt, "sma": sma, "stddev": stddev, "supertrend": supertrend, "t3": t3,
                    "trima": trima, "true_range": true_range, "wma": wma, "tema": tema, "chop": chop,
                    "atr": atr, "ao": ao, "dm": dm, "dema": dema, "bias": bias, "cfo": cfo, "cti": cti,
                    "inertia": inertia, "ppo": ppo, "trix": trix, "alma": alma, "fwma": fwma, "hilo": hilo,
                    "hl2": hl2, "hwma": hwma, "midpoint": midpoint, "midprice": midprice, "pwma": pwma,
                    "sinwma": sinwma, "kurtosis": kurtosis, "mad": mad, "skew": skew, "variance": variance
                }

                indicator_df = pd.DataFrame()
                for indicator_name, indicator_series in indicator_map.items():
                    indicator_series = indicator_series

                    # Normalize the indicator
                    normalized_indicator = indicator_series - indicator_series.rolling(rolling_window).mean()

                    # Add to the DataFrame
                    indicator_df[indicator_name] = normalized_indicator

                indicator_row = indicator_df.to_numpy()[-1, :]
                indicator_row = np.concatenate(([matrix[time, 0]], indicator_row))

                if indicator_matrix is None:
                    indicator_matrix = indicator_row
                else:
                    indicator_matrix = np.vstack([indicator_matrix, indicator_row])


        indicator_matrices[ticker] = indicator_matrix

        np.save(f'Indicator_Matrices/{ticker}_indicator_matrix.npy', indicator_matrix)
        print(f'Finished saving indicator matrix of {ticker}')
        print(f"Shape of indicator_matrix just saved: {indicator_matrix.shape}")

    #np.savez(f'Indicator_Matrices/total_indicator_dictionary.npz', **indicator_matrices)
    #print('Saved indicator matrix dictionary')

    return indicator_matrices

add_indicators(ticker_data_dictionary)
#index: indicators 1 to n are indexed 0 to n-1 respectively
