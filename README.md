# Detecting anomaly in stock price data using autoencoder architecture (unsupervised learning)
The problem is to detect anomalous behavior in stock price data collected over the years, using an unsupervised learning method, that is using the autoencoder architecture

## Data Collection and Visualization

1. I downloaded the stock price data of GE from yahoo finance website
2. The data is read as a pandas dataframe, it has almost 15000 observations and 7 features
3. Used seaborn to visualize the data, that is, time vs the closing price
4. Divide the data into train and test sets

## Data pre-processing
1. Here the data 'Last' column is separated out as a new data frame and is converted to a numpy array
2. The data is then scaled (I have used StandardScaler, that standardizes features by removing the mean and scaling to unit variance)
3. I used a sequence size of 60 here and then converted the observations to sequences with features and labels for both train and test sets

## Model building 
1. I used autoencoder model architecture
2. Model is compiled using the default values and mean absolute error loss is calculated
3. The model is trained with the data and the label with a batch_size of 32 and 50 epochs

# Plotting
1. The training and validation losses are plotted using matplotlib
2. We know that anomaly is detected where the reconstruction error is large and we can define a value beyond which we call it an anomaly
3. Looking at the MAE in training prediction
4. Gathering all details in a dataframe and using seaborn to do a lineplot of the same
5. Also plotted the anomalies as colored dots on the data

