from sklearn.neural_network import MLPRegressor


class MLPMidtermProject():
    def __init__(self, config):
        activation = config['activation']
        solver = config['solver']
        alpha = float(config['alpha'])
        batch_size = int(config['batch_size'])
        learning_rate = config['learning_rate']
        learning_rate_init = float(config['learning_rate_init'])
        max_iter = int(config['max_iter'])
        shuffle = config['shuffle'] in ['True']
        early_stopping = config['early_stopping'] in ['True']
        validation_fraction = float(config['validation_fraction'])
        n_iter_no_change = float(config['n_iter_no_change'])

        self.configuration = ',' + str(activation) + ',' + str(solver) + ',' + str(alpha) + ',' + str(batch_size) + ',' + str(learning_rate) + ',' + str(learning_rate_init) + ',' + str(max_iter) + ',' + str(shuffle) + ',' + str(early_stopping) + ',' + str(validation_fraction) + ',' + str(n_iter_no_change)

        self.mlp = MLPRegressor(hidden_layer_sizes=(10000), activation=activation, solver=solver, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter, shuffle=shuffle, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change)
    
    def train(self, x, y):
        self.mlp.fit(x,y)

    def predict(self, x):
        y_predicted = self.mlp.predict(x)
        return y_predicted

    def get_configuration(self,):
        return self.configuration