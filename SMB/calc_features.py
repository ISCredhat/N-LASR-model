import pandas as pd
import pandas_ta as ta


def calc_mean(data_series):
    return data_series
    # return data_series - data_series.rolling(window=14).mean()


def calc_daily_indicators(p):
    indicators = pd.DataFrame(index=p.index)
    indicators.index = p.index

    indicators['adx'] = calc_mean(ta.adx(p.high, p.low, p.close).ADX_14)  # returns pd.DataFrame: adx, dmp, dmn
    indicators['alma'] = calc_mean((ta.alma(p.close) - p.open) / p.open)
    indicators['ao'] = calc_mean((ta.ao(p.high, p.low) - p.open) / p.open)
    indicators['atr'] = calc_mean((ta.atr(p.high, p.low, p.close) - p.open) / p.open)
    indicators['bias'] = calc_mean((ta.bias(p.close) - p.open) / p.open)
    indicators['bop'] = calc_mean(ta.bop(p.open, p.high, p.low, p.close))
    indicators['cci'] = calc_mean(ta.cci(p.high, p.low, p.close))
    indicators['cfo'] = calc_mean((ta.cfo(p.close) - p.open) / p.open)
    indicators['chop'] = calc_mean((ta.chop(p.high, p.low, p.close) - p.open) / p.open)
    indicators['cmf'] = calc_mean(ta.cmf(p.high, p.low, p.close, p.volume))
    indicators['cmo'] = calc_mean(ta.cmo(p.close))
    indicators['corr'] = calc_mean(p.close.rolling(window=14).corr(p.close.shift(1)))
    indicators['cti'] = calc_mean((ta.cti(p.close) - p.open) / p.open)
    indicators['dm'] = calc_mean((ta.dm(p.high, p.low).iloc[:, 0] - p.open) / p.open)
    indicators['dema'] = calc_mean((ta.dema(p.close) - p.open) / p.open)
    indicators['dpo'] = calc_mean(ta.dpo(p.close))
    indicators['efi'] = calc_mean((ta.efi(p.close, p.volume) - p.open) / p.open)
    indicators['eri_bull'] = calc_mean((ta.eri(p.high, p.low, p.close).BULLP_13 - p.open) / p.open)
    indicators['eri_bear'] = calc_mean((ta.eri(p.high, p.low, p.close).BEARP_13 - p.open) / p.open)
    indicators['ema'] = calc_mean((ta.ema(p.close) - p.open) / p.open)
    indicators['eom'] = calc_mean(ta.eom(p.high, p.low, p.close, p.volume))
    indicators['fisher'] = calc_mean(ta.fisher(p.high, p.low).iloc[:, 0].astype('float64'))
    indicators['fwma'] = calc_mean((ta.fwma(p.close) - p.open) / p.open)
    indicators['hl2'] = calc_mean((ta.hl2(p.high, p.low) - p.open) / p.open)
    indicators['hilo'] = calc_mean((ta.hilo(p.high, p.low, p.close).iloc[:, 0] - p.open) / p.open)
    indicators['hma'] = calc_mean((ta.hma(p.close) - p.open) / p.open)
    indicators['hwma'] = calc_mean((ta.hwma(p.close) - p.open) / p.open)
    indicators['inertia'] = calc_mean((ta.inertia(p.close) - p.open) / p.open)
    indicators['kama'] = calc_mean(ta.kama(p.close))
    # indicators['kc_upper'] = kc_upper - kc_upper.rolling(rolling_window).mean()
    # pd.DataFrame: lower, basis, upper columns.
    # this is the wrong way to use this - should be close wrt the upper or lower
    indicators['kc_upper'] = calc_mean(ta.kc(p.high, p.low, p.close).KCUe_20_2)
    # the function returns pd.DataFrame: kst and kst_signal columns so .kst MAY NOT be correct?
    indicators['kst'] = calc_mean(ta.kst(p.close)['KST_10_15_20_30_10_10_10_15'])
    indicators['kurtosis'] = calc_mean((ta.kurtosis(p.close) - p.open) / p.open)
    indicators['linreg'] = calc_mean((ta.linreg(p.close) - p.open) / p.open)
    indicators['macd'] = calc_mean((ta.macd(p.close).iloc[:, 0] - p.open) / p.open)
    indicators['median'] = calc_mean((ta.median(p.close) - p.open) / p.open)
    indicators['midpoint'] = calc_mean((ta.midpoint(p.close) - p.open) / p.open)
    indicators['midprice'] = calc_mean((ta.midprice(p.high, p.low) - p.open) / p.open)
    indicators['mom'] = calc_mean((ta.mom(p.close) - p.open) / p.open)
    indicators['nvi'] = calc_mean((ta.nvi(p.close, p.volume) - p.open) / p.open)
    indicators['obv'] = calc_mean((ta.obv(p.close, p.volume) - p.open) / p.open)
    indicators['percent_return'] = calc_mean(ta.percent_return(p.close))
    indicators['ppo'] = calc_mean((ta.ppo(p.close).iloc[:, 0] - p.open) / p.open)
    indicators['pvi'] = calc_mean((ta.pvi(p.close, p.volume) - p.open) / p.open)
    indicators['pvo'] = calc_mean((ta.pvo(p.volume).iloc[:, 0] - p.open) / p.open)
    indicators['pvt'] = calc_mean((ta.pvt(p.close, p.volume) - p.open) / p.open)
    indicators['pwma'] = calc_mean((ta.pwma(p.close) - p.open) / p.open)
    indicators['mad'] = calc_mean((ta.mad(p.close) - p.open) / p.open)
    indicators['roc'] = calc_mean((ta.roc(p.close) - p.open) / p.open)
    indicators['rsi'] = calc_mean(ta.rsi(p.close))
    indicators['sinwma'] = calc_mean((ta.sinwma(p.close) - p.open) / p.open)
    indicators['skew'] = calc_mean((ta.skew(p.close) - p.open) / p.open)
    indicators['slope'] = calc_mean((ta.slope(p.close) - p.open) / p.open)
    indicators['sma'] = calc_mean((ta.sma(p.close) - p.open) / p.open)
    indicators['stddev'] = calc_mean((ta.stdev(p.close) - p.open) / p.open)
    indicators['supertrend'] = calc_mean((ta.supertrend(p.high, p.low, p.close).iloc[:, 0] - p.open) / p.open)
    indicators['t3'] = calc_mean((ta.t3(p.close) - p.open) / p.open)
    indicators['tema'] = calc_mean((ta.tema(p.close) - p.open) / p.open)
    indicators['trima'] = calc_mean((ta.trima(p.close) - p.open) / p.open)
    indicators['trix'] = calc_mean((ta.trix(p.close).iloc[:, 0] - p.open) / p.open)
    indicators['true_range'] = calc_mean((ta.true_range(p.high, p.low, p.close) - p.open) / p.open)
    indicators['uo'] = calc_mean(ta.uo(p.high, p.low, p.close))
    indicators['variance'] = calc_mean((ta.variance(p.close) - p.open) / p.open)
    # indicators['vortex'] = vortex - vortex.rolling(rolling_window).mean()
    # this is the wrong way to use this - should be close wrt the upper or lower?
    # https: // stockcharts.com / school / doku.php?id = chart_school:technical_indicators: vortex_indicator
    indicators['vortex_VTXP_14'] = calc_mean(ta.vortex(p.high, p.low, p.close).VTXP_14)
    indicators['vortex_VTXM_14'] = calc_mean(ta.vortex(p.high, p.low, p.close).VTXM_14)
    indicators['wma'] = calc_mean((ta.wma(p.close) - p.open) / p.open)
    indicators['willr'] = calc_mean(ta.willr(p.high, p.low, p.close))

    return indicators

def calc_all_stock_indicators(all_stock_data):
    all_stock_indicators_dict = {}
    for ticker in all_stock_data.columns.get_level_values(0).unique():
        indicators = calc_daily_indicators(all_stock_data[ticker])
        all_stock_indicators_dict[ticker] = indicators
        # print('Calculated indicators for:', ticker)

    combined_all_stock_indicators = pd.concat(all_stock_indicators_dict, axis=1)


    return combined_all_stock_indicators

