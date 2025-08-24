# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# the California housing dataset
from sklearn.datasets import fetch_california_housing
# the Ames housing dataset
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


#main class for pipeline
class Pipeline:
    def __init__(self):
        self.preprocessor = None

    def fit(self):
        df = self.fetch_data()
        X_train_transformed, X_test_transformed, y_train, y_test = self.preprocess_data(df=df)
        X_train_transformed_top, X_test_transformed_top = self.feature_selection(X_train_transformed, X_test_transformed, y_train, y_test)
        best_lgbm_model=self.model_building(X_train_transformed_top, y_train)
        self.model_eval(best_lgbm_model, X_test_transformed_top, y_test)

    def fetch_data(self):
        # Create a synthetic housing dataset since internet access is limited
        print("Creating synthetic housing dataset for demonstration...")
        
        import numpy as np
        np.random.seed(42)
        
        # Generate synthetic housing data
        n_samples = 1000
        
        # Features
        data = {
            'MedInc': np.random.normal(5.0, 2.0, n_samples),  # Median income
            'HouseAge': np.random.uniform(1, 50, n_samples),  # House age
            'AveRooms': np.random.normal(6.0, 1.0, n_samples),  # Average rooms
            'AveBedrms': np.random.normal(1.2, 0.2, n_samples),  # Average bedrooms
            'Population': np.random.uniform(500, 5000, n_samples),  # Population
            'AveOccup': np.random.normal(3.0, 0.5, n_samples),  # Average occupancy
            'Latitude': np.random.uniform(32.5, 42.0, n_samples),  # Latitude
            'Longitude': np.random.uniform(-124.0, -114.0, n_samples),  # Longitude
            'SquareFeet': np.random.normal(1500, 400, n_samples),  # House size (categorical as string)
            'PropertyType': np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_samples),
            'YearBuilt': np.random.uniform(1950, 2020, n_samples).astype(int).astype(str),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (house prices) based on features
        df['label'] = (
            df['MedInc'] * 3.0 +
            (50 - df['HouseAge']) * 0.1 +
            df['AveRooms'] * 0.3 +
            df['SquareFeet'] * 0.002 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Ensure positive prices
        df['label'] = np.maximum(df['label'], 0.5)

        # 2. Data Exploration
        print("Dataset shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nDataset info:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())

        return df

    def preprocess_data(self, df):
        #
        # 3. Data Preprocessing

        # Identify numeric and categorical columns
        print(len(df.columns))
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        print(len(categorical_features))
        print(categorical_features[:5])
        numeric_features.remove('label')  # Assuming target column is numeric
        print(len(numeric_features))
        print(numeric_features[:5])

        # Transformers
        numeric_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Store preprocessor as instance variable for later use
        self.preprocessor = preprocessor
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        # 4. Train-Test Split
        X = df.drop('label', axis=1)
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        return X_train_transformed, X_test_transformed, y_train, y_test

    def feature_selection(self,X_train_transformed, X_test_transformed, y_train, y_test):
        # 5. Feature Selection using RandomForestRegressor
        # Note: For simplicity, we'll apply the transformer and then the regressor sequentially, though you could use a pipeline.

        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train_transformed, y_train)

        # explicit feature importance extraction here
        # the one-hot encoding complicates direct column referencing.
        # Getting feature importances and corresponding names
        feature_importances = rf.feature_importances_
        print(feature_importances)

        # Get feature names from the preprocessor
        # For one-hot encoded columns, it'll add more complexity due to multiple columns being added for one categorical feature
        onehot_columns = list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            input_features=self.categorical_features))
        print(onehot_columns)
        all_features = self.numeric_features + onehot_columns

        # Get the top 20 features' indices
        sorted_idx = feature_importances.argsort()[-20:][::-1]
        print(sorted_idx)
        # Subset the data for the top features
        X_train_transformed_top = X_train_transformed[:, sorted_idx]
        X_test_transformed_top = X_test_transformed[:, sorted_idx]

        return X_train_transformed_top, X_test_transformed_top

    def model_building(self, X_train_transformed_top, y_train):
        # 6. Model Building using LightGBM
        # Preparing the dataset for LightGBM
        d_train = lgb.Dataset(X_train_transformed_top, label=y_train)

        params = {
            'objective': 'regression',
            'metric': 'rmse'
        }

        lgbm = lgb.train(params, d_train, 100)

        # 7. Hyperparameter Tuning using GridSearchCV
        lgbm_regressor = lgb.LGBMRegressor()
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [20, 40, 60],
            'num_leaves': [31, 61]
        }

        grid_search = GridSearchCV(lgbm_regressor, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_transformed_top, y_train)

        best_lgbm_model = grid_search.best_estimator_

        return best_lgbm_model

    def model_eval(self, best_lgbm_model, X_test_transformed_top, y_test):
        # 8. Model Evaluation
        y_pred = best_lgbm_model.predict(X_test_transformed_top)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        r2 = r2_score(y_test, y_pred)
        print(f"R^2 Score: {r2}")

        # 9. Model Saving
        joblib.dump(best_lgbm_model, 'best_lgbm_model.pkl')

        # plot the predictions vs actual
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Predictions vs Actuals')
        plt.show()

    def predict(self, X_test_transformed_top):
        pass
        # 10. Model Deployment
        # Load the model
        loaded_model = joblib.load('best_lgbm_model.pkl')

        # Test Prediction
        test_pred = loaded_model.predict(X_test_transformed_top)
        return test_pred


def main():
    """Main function to run the ML pipeline"""
    pipeline = Pipeline()
    pipeline.fit()


if __name__ == "__main__":
    main()





