import numpy as np
from scipy.stats import rankdata

tickers = ['TRMB', 'HBI', 'M', 'ARKQ', 'LYV', 'PRU', 'BBY']

total_quantiled_indicator_matrix = np.load(f'Indicator_Matrices/test_total_feature_matrix.npy', allow_pickle=True)
total_ticker_data = np.load(f'Indicator_Matrices/test_total_ticker_data.npy', allow_pickle=True)

#print(total_ticker_data[:2, :5, -3:])

threshold_var = 0.8
anomaly_threshold_var = 0

def features_with_ranked_FR(ticker_mat, quantiled_mat, threshold_value, anomaly_threshold_value):

    def cross_rank_FRs(mat1):
        ranked_FR_mat = np.empty((mat1.shape[1], mat1.shape[0]))

        for time in range(mat1.shape[1]):  # Loop over timesteps
            FR_values = mat1[:, time, -1]  # Extract FR values for all stocks at this timestep
            ranked_FR_values = (rankdata(FR_values) - 1) / (mat1.shape[0] - 1)  # Normalize ranks
            ranked_FR_mat[time, :] = ranked_FR_values  # Store the ranks

        # Reshape ranked_FR_mat to align with matrix dimensions
        ranked_FR_mat = ranked_FR_mat.T[..., np.newaxis]  # Shape: (num_stocks, timesteps, 1)

        # Concatenate the ranked FR matrix with the original matrix
        matrix_with_ranked_FR = np.concatenate((mat1, ranked_FR_mat), axis=2)

        return matrix_with_ranked_FR

    def label_FRs(mat1, mat2):
        top_threshold = threshold_value
        bottom_threshold = 1-threshold_value
        top_anomaly_threshold = 1-anomaly_threshold_value
        bottom_anomaly_threshold = anomaly_threshold_value

        # Initialize a labels array with the same shape as the ranked_FR_values
        labels = np.zeros((mat1.shape[0], mat1.shape[1]), dtype=int)

        # Assign labels based on thresholds
        for stock in range(mat1.shape[0]):
            for time in range(mat1.shape[1]):
                ranked_FR_values = mat1[stock, time, -1]  # Access the FR value for this stock and time
                if top_threshold <= ranked_FR_values <= top_anomaly_threshold:
                    labels[stock, time] = 1  # Assign 1 for top threshold range
                elif bottom_anomaly_threshold <= ranked_FR_values <= bottom_threshold:
                    labels[stock, time] = -1  # Assign -1 for bottom threshold range

        # Concatenate the labels as a new feature dimension
        labeled_matrix = np.concatenate((mat2, labels[..., np.newaxis]), axis=2)

        return labeled_matrix

    tickers_with_ranked_FR = cross_rank_FRs(ticker_mat)
    features_with_labeled_FR = label_FRs(tickers_with_ranked_FR, quantiled_mat)

    return features_with_labeled_FR

features_and_FRs = features_with_ranked_FR(total_ticker_data, total_quantiled_indicator_matrix, threshold_var, anomaly_threshold_var)
print('features_and_FRs shape', features_and_FRs.shape)

#example numbers below, the function is for use within a rolling window function to test day by day predictions
training_window = 37
window_start = 0
window_end = window_start + training_window

def create_training_and_testing_matrix(matrix, window_end, window_start):
    matrix_window = matrix[:, window_start:window_end+1, :]
    FR_labeled_column = matrix_window[:, :, -1]
    FR_mask = (FR_labeled_column != 0)

    training_matrix = matrix_window[FR_mask]
    training_features = training_matrix[:, :-1]
    training_labels = training_matrix[:, -1]

    testing_matrix = matrix[:, window_end+1, :-1]

    return training_features, training_labels, testing_matrix

training_features,training_labels, testing_matrix = create_training_and_testing_matrix(features_and_FRs, window_end, window_start)

print('training features matrix shape', training_features.shape)
print('training labels array shape', training_labels.shape)
print('testing mat shape', testing_matrix.shape)










