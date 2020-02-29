from sklearn.neural_network import MLPRegressor


class MLPMidtermProject():
    def __init__(self, config):
        hidden_layer_size = int(config['hidden_layer_size'])
        activation = config['activation']
        solver = config['solver']
        alpha = int(config['alpha'])
        batch_size = int(config['batch_size'])
        learning_rate = config['learning_rate']
        learning_rate_init = config['learning_rate_init']
        max_iter = int(config['max_iter'])
        shuffle = bool(config['shuffle'])
        early_stopping = bool(config['early_stopping'])
        validation_fraction = float(config['validation_fraction'])
        n_iter_no_change = float(config['n_iter_no_change'])
        
        self.mlp = MLPRegressor((hidden_layer_sizes=(100, ), activation=activation, solver=solver, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter, shuffle=shuffle, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change))
    
    def train(self, x, y):
        self.mlp.fit(x,y)

    def predict(self, x):
        y_predicted = self.mlp.predict(x)
        return y_predicted





        