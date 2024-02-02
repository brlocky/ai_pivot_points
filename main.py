import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf
from pandas import Series
import numpy as np
import pickle


class MyProgram:
    def __init__(self):
        # Load the saved model from file
        try:
            # Load the saved model from file
            self.model = pickle.load(open('model.pkl', 'rb'))
            print("Model loaded.")
        except FileNotFoundError:
            print("Creating new model.")
            self.model = self.create_model()

    def create_model(self):
        # Define the model architecture
        hidden_units = 512

        # Create a placeholder for the input shape
        inputs = tf.keras.Input(shape=(None,))

        # Reshape the input to add a third dimension
        reshaped_inputs = tf.expand_dims(inputs, axis=-1)

        # Create the LSTM layer
        lstm_layer = tf.keras.layers.LSTM(hidden_units)(reshaped_inputs)

        # Create two output layers with sigmoid activation for binary classification
        output_top = tf.keras.layers.Dense(
            1, activation='sigmoid', name='output_top')(lstm_layer)
        output_bottom = tf.keras.layers.Dense(
            1, activation='sigmoid', name='output_bottom')(lstm_layer)

        # Create the model with two outputs
        model = tf.keras.Model(inputs=inputs, outputs=[output_top, output_bottom])

        learning_rate = 0.008  # Set your desired learning rate here
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['mse'])

        # Print the model summary
        # model.summary()
        return model

    def load_df(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = [column.capitalize() for column in df.columns]
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        return df

    def load_train_data(self, file_path):
        df = self.load_df(file_path)
        data = pd.DataFrame(df)
        y_test_top = np.array(data['Pivot'].apply(lambda x: 1 if x > 0 else 0))
        y_test_bottom = np.array(data['Pivot'].apply(
            lambda x: 1 if x == -1 or x == 3 else 0))

        predictions = [y_test_top, y_test_bottom]
        self.plot_chart(df, predictions, file_path + '_original.png')

        # Clean Data
        data = data.drop(['Pivot', 'Volume'], axis=1)

        train_X = self.scale_data(data)
        train_y = [y_test_top, y_test_bottom]
        return train_X, train_y

    def load_predict_data(self, file_path):
        df = self.load_df(file_path)
        data = pd.DataFrame(df)
        # Clean Data
        data = data.drop(['Volume'], axis=1)
        try:
            data = data.drop(['Pivot'], axis=1)
        except Exception:
            print('not found pivot')

        return self.scale_data(data)

    def scale_data(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        return scaled_data

    def train_model(self, train_X, train_y):
      # Train your model
        history = self.model.fit(train_X, train_y, epochs=32,
                                 batch_size=len(train_X), verbose=0)
        print(history.history['loss'][-1])
        print(history.history['output_top_mse'][-1])
        print(history.history['output_bottom_mse'][-1])

    def predict_model(self, test_X):
        predictions = self.model.predict(test_X)
        predictions_array = np.array(predictions)
        return (predictions_array > 0.5).astype(int)

    def plot_chart(self, df, predictions, output):
        style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={
                                   'figure.facecolor': 'white'})

        def plot_crosses(df, predictions):
            crosses = []
            pivot_points_high = [np.nan] * len(df)
            pivot_points_low = [np.nan] * len(df)

            for i, output_predictions in enumerate(predictions):

                is_top_pivots = 1 if i == 0 else 0

                for j, (index, row) in enumerate(df.iterrows()):
                    to_mark = 1 if output_predictions[j] == 1 else 0
                    if to_mark:
                        if is_top_pivots == 1:
                            pivot_points_high[j] = row['High']
                        else:
                            pivot_points_low[j] = row['Low']

            if not Series(pivot_points_high).isna().all():
                crosses.append(mpf.make_addplot(
                    pivot_points_high,
                    type="scatter",
                    color="green",
                    marker="x",
                    alpha=0.7,
                    markersize=50
                ))

            if not Series(pivot_points_low).isna().all():
                crosses.append(mpf.make_addplot(
                    pivot_points_low,
                    type="scatter",
                    color="red",
                    marker="x",
                    alpha=0.7,
                    markersize=50
                ))

            return crosses

        # Plot the candlestick chart with crosses
        fig, axlist = mpf.plot(df, type='ohlc', style=style, show_nontrading=False,
                               title='Candlestick Chart', addplot=plot_crosses(df, predictions),
                               returnfig=True)

        plt.savefig(output)
        # Show the plot
        # plt.show()
        # Optionally, close the figure to release resources
        plt.close(fig)

    def run_train(self, file_path):
        # Your code here
        train_X, train_y = self.load_train_data(file_path)
        self.train_model(train_X, train_y)
        pickle.dump(self.model, open('model.pkl', 'wb'))

    def run_learn(self):
        folder_path = 'csv/learn'  # Path to the 'learn' folder

        file_names = os.listdir(folder_path)
        random.shuffle(file_names)  # Shuffle the list of file names

        for file_name in file_names:
            if file_name.endswith('.csv'):  # Check if the file is a CSV file
                # Get the full path of the CSV file
                file_path = os.path.join(folder_path, file_name)
                train_X, train_y = self.load_train_data(file_path)
                self.train_model(train_X, train_y)
                pickle.dump(self.model, open('model.pkl', 'wb'))

    def run_predict_all(self):
        folder_path = 'csv/test'  # Path to the 'learn' folder

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # Check if the file is a CSV file
                # Get the full path of the CSV file
                file_path = os.path.join(folder_path, file_name)
                test_X = self.load_predict_data(file_path)
                predictions = self.predict_model(test_X)
                df = self.load_df(file_path)
                self.plot_chart(df, predictions, file_path + '.png')
                accuracy = self.evaluate_predict(df, predictions)
                print(f"Accuracy: {accuracy * 100}%")

    def evaluate_predict(self, df, predictions):
        predictionTop = predictions[0]
        predictionBottom = predictions[1]

        pivots = df['Pivot']
        correct_predictions = 0

        for i in range(0, len(pivots)):
            pivot = pivots.iloc[i]
            top = predictionTop[i]
            bottom = predictionBottom[i]

            if pivot == 0:
                if top == 0 and bottom == 0:
                    correct_predictions += 1

            if pivot == 1:
                if top == 1 and bottom == 0:
                    correct_predictions += 1

            if pivot == -1:
                if top == 0 and bottom == 1:
                    correct_predictions += 1

            if pivot == 3:
                if top == 1 and bottom == 1:
                    correct_predictions += 1

        accuracy = correct_predictions / len(pivots)
        return accuracy

    def run(self):
        print('Ready')


if __name__ == "__main__":
    program = MyProgram()
    program.run()
