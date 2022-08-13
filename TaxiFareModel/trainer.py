# imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

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
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipeline

    def run(self):
        """set and train the pipeline"""
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    
    # get data
    df = get_data()
        
    # clean data
    df_clean = clean_data(df)
        
    # set X and y
    y = df_clean["fare_amount"]
    X = df_clean.drop("fare_amount", axis=1)
        
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train

    trainer = Trainer(X_train, y_train)
    pipeline = trainer.set_pipeline()
    trainer.run()
        
    # evaluate
    rmse = trainer.evaluate(X_val, y_val)
        
    #print('TODO')