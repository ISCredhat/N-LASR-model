import numpy as np
from scipy.stats import rankdata


def features_with_ranked_FR(ticker_mat, quantiled_mat, threshold_value, anomaly_threshold_value):

    #cross rank FR values per timestep across all stocks
    def cross_rank_FRs(mat1):
        ranked_FR_mat = np.empty((mat1.shape[1], mat1.shape[0]))

        for time in range(mat1.shape[1]):  # Loop over timesteps
            FR_values = mat1[:, time, -1]  # Extract FR values for all stocks at this timestep

            #I believe this will rank my higher FR's as a higher value, i.e. 5% would rank 125 if best return, while 4.5% would rank 124 at second best
            #therefore the better forward return values are close to 1
            ranked_FR_values = (rankdata(FR_values) - 1) / (mat1.shape[0] - 1)  # Normalize ranks
            ranked_FR_mat[time, :] = ranked_FR_values  # Store the ranks

        # Reshape ranked_FR_mat to align with matrix dimensions
        ranked_FR_mat = ranked_FR_mat.T[..., np.newaxis]  # Shape: (num_stocks, timesteps, 1)

        # Concatenate the ranked FR matrix with the original matrix
        matrix_with_ranked_FR = np.concatenate((mat1, ranked_FR_mat), axis=2)

        return matrix_with_ranked_FR

    def label_FRs(mat1, mat2):
        #due to the fact that my better FR's are close to 1, my threshold must be greater than 0.5
        top_threshold = threshold_value

        #want anomaly threshold value to be between 0 and 0.1 (these are what I assume to be likely values, could change)
        #this results in anomaly threshold between 0.9 and 1
        anomaly_threshold = 1-anomaly_threshold_value

        labels = np.zeros((mat1.shape[0], mat1.shape[1]), dtype=int)

        # Assign labels based on thresholds
        for stock in range(mat1.shape[0]):
            for time in range(mat1.shape[1]):
                ranked_FR_values = mat1[stock, time, -1]  # Access the FR value for this stock and time
                if top_threshold <= ranked_FR_values <= anomaly_threshold:
                    labels[stock, time] = 1  # Assign 1 for top threshold range
        # Concatenate the labels as a new feature dimension
        labeled_matrix = np.concatenate((mat2, labels[..., np.newaxis]), axis=2)

        return labeled_matrix

    tickers_with_ranked_FR = cross_rank_FRs(ticker_mat)
    features_with_labeled_FR = label_FRs(tickers_with_ranked_FR, quantiled_mat)

    return features_with_labeled_FR


def create_training_and_testing_matrix(matrix, window_end, window_start):
    matrix_window = matrix[:, window_start:window_end+1, :]
    FR_labeled_column = matrix_window[:, :, -1]
    FR_mask = (FR_labeled_column != 0)

    training_matrix = matrix_window[FR_mask]
    training_features = training_matrix[:, :-1]
    training_labels = training_matrix[:, -1]

    testing_matrix = matrix[:, window_end+1, :-1]

    return training_features, training_labels, testing_matrix










