#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class FeatureEngineer:
    def __init__(self, data):
        self.data = data
    
    def drop_columns(self, columns):
        """
        Drop specified columns from the DataFrame.
        
        Args:
            columns (list): List of column names to drop.
        
        Returns:
            DataFrame: The DataFrame with the specified columns dropped.
        """
        try:
            self.data = self.data.drop(columns=columns)
        except:
            print("Columns already deleted")
    
    def convert_to_datetime(self, column_name):
        """
        Convert a specified column to a datetime format.
        
        Args:
            column_name (str): Name of the column to convert.
        
        Returns:
            DataFrame: The DataFrame with the specified column converted to datetime.
        """
        self.data[column_name] = pd.to_datetime(self.data[column_name])
    
    def create_target_column(self, result_column_name, target_column_name):
        """
        Create a target column based on the values in a specified result column.
        
        Args:
            result_column_name (str): Name of the column containing the result values.
            target_column_name (str): Name of the target column to create.
        
        Returns:
            DataFrame: The DataFrame with the target column created based on the values in the result column.
        """
        self.data[target_column_name] = (self.data[result_column_name] == "W").astype("int")
    
    def create_venue_code_column(self, venue_column_name):
        """
        Create a new column containing codes for the unique values in a specified venue column.
        
        Args:
            venue_column_name (str): Name of the venue column to convert to codes.
        
        Returns:
            DataFrame: The DataFrame with a new column containing codes for the unique values in the venue column.
        """
        self.data["venue_code"] = self.data[venue_column_name].astype("category").cat.codes
    
    def create_opp_code_column(self, opp_column_name):
        """
        Create a new column containing codes for the unique values in a specified opponent column.
        
        Args:
            opp_column_name (str): Name of the opponent column to convert to codes.
        
        Returns:
            DataFrame: The DataFrame with a new column containing codes for the unique values in the opponent column.
        """
        self.data["opp_code"] = self.data[opp_column_name].astype("category").cat.codes
    
    def create_hour_column(self, time_column_name):
        """
        Create a new column containing hour values extracted from a specified time column.
        
        Args:
            time_column_name (str): Name of the time column to extract hour values from.
        
        Returns:
            DataFrame: The DataFrame with a new column containing hour values extracted from the time column.
        """
        self.data["hour"] = self.data[time_column_name].str.replace(":.+", "", regex=True).astype("int")
    
    def create_day_code_column(self, date_column_name):
        """
        Create a new column containing codes for the days of the week in a specified date column.
        
        Args:
            date_column_name (str): Name of the date column to extract day codes from.
        
        Returns:
            DataFrame: The DataFrame with a new column containing day codes extracted from the date column.
        """
        self.data["day_code"] = self.data[date_column_name].dt.dayofweek
        
    def create_baseline_features(self):
        """
        Perform feature engineering on the data by dropping irrelevant
        Returns:
            DataFrame: The preprocessed DataFrame with relevant columns dropped and new feature columns added.
        """
        self.drop_columns(["comp", "notes"])
        self.convert_to_datetime("Date")
        self.create_target_column("Result", "target")
        self.create_venue_code_column("Venue")
        self.create_opp_code_column("Opponent")
        self.create_hour_column("Time")
        self.create_day_code_column("Date")
        return ["venue_code", "opp_code", "hour", "day_code"]
        
        
    
    
    
    def rolling_averages(self, group, cols, new_cols):
        """
        Create rolling averages for specified columns in a group.
        
        Args:
            group (DataFrame): The group to calculate rolling averages for.
            cols (list): A list of column names to calculate rolling averages for.
            new_cols (list): A list of column names to use for the rolling averages.
        
        Returns:
            DataFrame: The original group DataFrame with the rolling averages added as new columns.
        """
        group = group.sort_values("Date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group
    
    def create_rolling_average_columns(self):
        """
        Create rolling average columns for specified columns in the DataFrame.
        """
        cols =['GF', 'GA', "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
        new_cols = [f"{c}_rolling" for c in cols]

        matches_rolling = self.data.groupby("Team").apply(lambda x: self.rolling_averages(x, cols, new_cols))
        matches_rolling = matches_rolling.droplevel('Team')

        self.data[new_cols] = matches_rolling[new_cols]
        return new_cols
        
        
    

    def test_features(self, start_date: str, end_date: str, features: List[str]) -> float:
        """
        Trains a random forest classifier on the selected features using the matches data between
        start_date and end_date, and returns the accuracy score on the test data.

        Parameters:
        start_date (str): Start date (inclusive) for selecting the matches data
        end_date (str): End date (exclusive) for selecting the matches data
        features (List[str]): List of features to be used for training the model

        Returns:
        float: Accuracy score of the random forest classifier on the test data
        """
        
        # Select the matches data between start_date and end_date
        train = self.data[self.data["Date"] < end_date]
        test = self.data[self.data["Date"] >= start_date]

        # Train the random forest classifier on the selected features
        rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
        rf.fit(train[features], train["target"])

        # Predict the target variable on the test data
        preds = rf.predict(test[features])

        # Calculate the accuracy score of the model
        acc = accuracy_score(test["target"], preds)

        # Return the accuracy score
        return acc



# In[153]:


if __name__ == "__main__":
    
    data = pd.read_csv("matches.csv")
    import DataPreprocessor as dp

    data = dp.DataPreprocessing(data).process_data()

    ## Creating features

    feature_engineer = FeatureEngineer(data)
    
    base_line_features = feature_engineer.create_baseline_features()
    new_cols = feature_engineer.create_rolling_average_columns()
    preprocessed_data = feature_engineer.data
    feature_engineer.data  = preprocessed_data = preprocessed_data.dropna()

    ## Testing features

    start_date = '2022-01-01'
    end_date = '2022-02-01'
    accuracy = feature_engineer.test_features(start_date, end_date, base_line_features+new_cols)

    print(f"Accuracy score: {accuracy:.4f}")


# In[ ]:




