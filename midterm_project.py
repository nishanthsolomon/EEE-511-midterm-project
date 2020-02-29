import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

from mlp import MLPMidtermProject
from linear_regression import LinearRegressionMidtermProject

class Midterm():
    def __init__(self, config):
        data_config = config['data']
        mlp_config = config['mlp']

        train_data = pd.read_csv(data_config['train_data_path'], delimiter = data_config['delimiter'])
        test_data = pd.read_csv(data_config['test_data_path'], delimiter = data_config['delimiter'])

        output_label = data_config['output_label']
        feature_labels = data_config['feature_labels'].split(',')

        self.y_train = train_data[output_label]
        self.x_train = train_data[feature_labels]

        self.y_test = test_data[output_label]
        self.x_test = test_data[feature_labels]

        self.linear_regression = LinearRegressionMidtermProject()
        self.mlp = MLPMidtermProject(mlp_config)
        
    def run_linear_regression(self):
        self.linear_regression.train(self.x_train, self.y_train)
        y_predicted = self.linear_regression.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_predicted)
        return rmse
    
    def run_polinomial_regression(self):
        new_x = self.x_train
        new_x['Temperature AVG square'] = new_x['Temperature AVG'] * new_x['Temperature AVG']
        new_x['Relative Humidity AVG square'] = new_x['Relative Humidity AVG'] * new_x['Relative Humidity AVG']
        new_x['Wind Speed Daily AVG square'] = new_x['Wind Speed Daily AVG'] * new_x['Wind Speed Daily AVG']
        new_x_test = self.x_test
        new_x_test['Temperature AVG square'] = new_x_test['Temperature AVG'] * new_x_test['Temperature AVG']
        new_x_test['Relative Humidity AVG square'] = new_x_test['Relative Humidity AVG'] * new_x_test['Relative Humidity AVG']
        new_x_test['Wind Speed Daily AVG square'] = new_x_test['Wind Speed Daily AVG'] * new_x_test['Wind Speed Daily AVG']
        self.linear_regression.train(new_x, self.y_train)

        y_predicted = self.linear_regression.predict(new_x_test)
        rmse = self.rmse(self.y_test, y_predicted)
        return rmse

    def run_mlp(self):
        self.mlp.train(self.x_train, self.y_train)
        y_predicted = self.mlp.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_predicted)
        return rmse

    def rmse(self, y, y_predicted):
        self.plot_results(y_predicted)
        rms = sqrt(mean_squared_error(y, y_predicted))
        return rms

    def plot_results(self, y_predicted):
        X = np.array(self.x_test)
        Y = np.array(self.y_test)
        Y_predicted = np.array(y_predicted)

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Y, color='r', label='Actual Success Rate')
        ax.scatter(X[:, 0], X[:, 1], Y_predicted, color='g', label='Predicted Success Rate')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./midterm_project.conf')

    midterm_project = Midterm(config)

    rmse_linear_regression = midterm_project.run_linear_regression()
    rmse_polynomial_regression = midterm_project.run_polinomial_regression()
    rmse_mlp = midterm_project.run_mlp()


    print('RMS error of linear regression model = ' + str(rmse_linear_regression))
    print('RMS error of polynomial regression model = ' + str(rmse_polynomial_regression))
    with open('training.csv', 'a') as training_file:
        training_file.write('100 1000 100 25' + midterm_project.mlp.get_configuration() + ',' + str(rmse_mlp) + '\n')
    print('RMS error of mlp model = ' + str(rmse_mlp))
