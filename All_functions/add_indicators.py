import pandas as pd
import numpy as np
import pandas_ta as ta


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
                adx = ta.adx(high_col, low_col, close_col)
                rsi = ta.rsi(close_col)
                uo = ta.uo(high_col, low_col, close_col)
                willr = ta.willr(high_col, low_col, close_col)
                kc_upper = ta.kc(high_col, low_col, close_col)
                kama = ta.kama(close_col)
                vtx = ta.vortex(high_col, low_col, close_col)
                cmf = ta.cmf(high_col, low_col, close_col, volume_col)
                kst = ta.kst(close_col)
                bop = ta.bop(open_col, high_col, low_col, close_col)
                cci = ta.cci(high_col, low_col, close_col)
                cmo = ta.cmo(close_col)
                percent_return = ta.percent_return(close_col)

                # non-normal indicators
                efi = (ta.efi(close_col, volume_col) - open_col) / open_col
                eri = (ta.eri(high_col, low_col, close_col) - open_col) / open_col
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

                if indicator_matrix is None:
                    indicator_matrix = indicator_row
                else:
                    np.vstack([indicator_matrix, indicator_row])


        indicator_matrices[ticker] = indicator_matrix

    return indicator_matrices

#index: indicators 1 to n are indexed 0 to n-1 respectively
