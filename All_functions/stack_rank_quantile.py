import numpy as np
from scipy.stats import rankdata

num_stocks = 125

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








