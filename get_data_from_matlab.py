import scipy.io
import pandas as pd

# Define the file path
file_path = '/Users/stephanie/Downloads/AAPL.mat'

# Load the .mat file
mat_contents = scipy.io.loadmat(file_path)

# Display the contents
print(mat_contents)

data = mat_contents['data']

# Assuming 'data' is your variable with mat contents
# Let's assume 'data' is a numpy array for this example
# data = ...

# Create a DataFrame from 'data'
df = pd.DataFrame(data)

# Convert the first column to datetime and set it as the index
df[0] = pd.to_datetime(df[0], errors='coerce')  # Convert to datetime
df.set_index(0, inplace=True)  # Set first column as index

# Rename the columns
df.columns = ['close', 'volume', 'num_trades']

# Output the resulting DataFrame
print(df)