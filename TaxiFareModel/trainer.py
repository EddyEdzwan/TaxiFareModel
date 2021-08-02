# imports
import mlflow
import joblib
from  mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    '''
    Main class object to train a regression model on the Kaggle Taxi Fare Dataset 
    '''
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    experiment_name = "[SG] [SG] [EddyEdzwan] Simple Linear + 1.0"
    
    def __init__(self, X, y):
        """
            INSTANTIATE WITH TRAINING DATA ONLY
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self, model='linear', **kwargs):
        """
        Create the pipeline and save it as a class attribute
        """

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

        #Create final pipeline with model
        if model == 'linear':
            final_pipe = Pipeline([
                ('preproc', preproc),
                ('model', LinearRegression())
            ])

        if model == 'knn':
            final_pipe = Pipeline([
                ('preproc', preproc),
                ('model', KNeighborsRegressor())
            ])

        if model == 'svr':
            final_pipe = Pipeline([
                ('preproc', preproc),
                ('model', SVR())
            ])

        if model == 'sgd':
            final_pipe = Pipeline([
                ('preproc', preproc),
                ('model', SGDRegressor())
            ])

        if model == 'lasso':
            final_pipe = Pipeline([
                ('preproc', preproc),
                ('model', Lasso())
            ])

        self.pipeline = final_pipe

        return final_pipe

    def run(self, model='linear', **kwargs):
        """
        Set and train the pipeline with a specified model [linear, knn, svr, sgd, lasso]
        Using Sklearn's instances of LinearRegression(), KNeighborsRegressor(), SVR(),
        SGDRegressor(), Lasso() respectively, with default parameters if none are passed
        
        Tuning the parameters using tune_model()
        """

        self.set_pipeline(model=model)

        if kwargs:
            self.pipeline.named_steps['model'].set_params(**kwargs)

        self.pipeline.fit(self.X, self.y)

        return self

    def tune_model(self, model='linear', **kwargs):
        """
        Returns the best model and best score achieved by cross-validation through
        Grid-Searching the parameters pass as arguments for the specific model
        Refer the sklearn documents for model hyperparameters to tune 
        """
        self.pipeline = self.set_pipeline(model=model)

        # Set up grid params
        grid_params = {}

        if kwargs:
            for key, value in kwargs.items():
                grid_params['model__' + key] = value
            
        # Instantiate CV
        search_results = GridSearchCV(self.pipeline, grid_params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

        # Fit data to CV
        search_results.fit(self.X, self.y)

        print('Best model obtained from Grid Search:')
        print(search_results.best_estimator_.named_steps['model'])

        return search_results.best_estimator_.named_steps['model'], search_results.best_score_

    def evaluate(self, X_test, y_test, **kwargs):
        """
        Evaluates the pipeline on cross validation and test data provided and return the RMSE
        Also evaluates on validation set, if passed
        """

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        self.mlflow_log_param('model/estimator', self.pipeline.named_steps['model'])
        self.mlflow_log_metric('rmse', rmse)

        cv_results = cross_validate(self.pipeline, self.X, self.y, cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error')

        cv_score = abs(cv_results['test_score'].mean())

        if kwargs:
            y_pred_val = self.pipeline.predict(kwargs['X_val'])
            val_rmse = compute_rmse(y_pred_val, kwargs['y_val'])
            self.mlflow_log_metric('val_rmse', val_rmse)
            return {'cv_rmse': cv_score, 'val_rmse':val_rmse, 'rmse': rmse}

        return {'cv_rmse': cv_score, 'rmse': rmse}

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        
        joblib.dump(self.pipeline,'model.joblib')
        pass

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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    score = trainer.evaluate(X_test, y_test, X_val=X_val, y_val=y_val)

    print(f'The model has a rmse score of {score}')
