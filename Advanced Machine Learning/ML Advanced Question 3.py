import dask.dataframe as dd
import dask_ml.model_selection as dcv
import pandas as pd
from dask_ml.linear_model import LinearRegression

data = pd.read_csv('advertising.csv')
# Create a Dask DataFrame from the dataset
df = dd.from_pandas(data, npartitions=1)  # Assuming 'data' contains the dataset

# Split the dataset into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = dcv.train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train.values, y_train.values)

# Predict sales using the test set
y_pred = model.predict(X_test.values)

# Compute the mean absolute error
mae = dd.compute(dd.metrics.mean_absolute_error(y_test, y_pred))

print("Mean Absolute Error:", mae[0])
