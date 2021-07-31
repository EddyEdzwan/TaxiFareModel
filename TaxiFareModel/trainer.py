# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        #Create pipelines for different features
        distance_pipe = Pipeline([
            ('haversine_calc', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('feature_extraction', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        #Combine the pipelines for pre-processing
        distance_features = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
        time_features = ['pickup_datetime']
        
        preproc = ColumnTransformer([
            ('distance_pipe', distance_pipe, distance_features),
            ('time_pipe', time_pipe, time_features)],
            remainder='drop')

        #Create final pipeline
        final_pipe = Pipeline([
            ('preproc', preproc),
            ('linear_model', LinearRegression())
        ])

        return final_pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()

        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)

        rmse = compute_rmse(y_pred, y_test)

        # print(rmse)

        return rmse


if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    data = clean_data(data)
    # set X and y
    y = data.pop('fare_amount')
    X = data
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    score = trainer.evaluate(X_test, y_test)
    print(f'The model has a rmse score of {score}')
