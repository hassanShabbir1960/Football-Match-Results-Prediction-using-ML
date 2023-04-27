#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, data):
        self.data = data
    
    def find_columns_with_same_values_or_all_nans(self):
        """
        Find columns with the same value for all rows or with all NaNs in a DataFrame.
        
        Returns:
            dict: A dictionary with column names as keys and their properties ("Same Values" or "All NaNs") as values.
        """
        # Calculate the number of unique values in each column
        unique_values = self.data.nunique(dropna=False)
        
        # Filter columns with only one unique value or all NaNs
        columns_to_check = unique_values[unique_values <= 1].index.tolist()
        
        column_properties = {}
        for column in columns_to_check:
            if self.data[column].isna().all():
                column_properties[column] = "All NaNs"
            else:
                column_properties[column] = "Same Values"
        
        return column_properties
    
    def drop_columns_with_same_values_or_all_nans(self):
        """
        Drop columns with the same value for all rows or with all NaNs in a DataFrame.
        
        Returns:
            DataFrame: The preprocessed DataFrame with the relevant columns removed.
        """
        column_info = self.find_columns_with_same_values_or_all_nans()
        columns_to_drop = list(column_info.keys())
        
        try:
            # Drop the columns from the DataFrame
            preprocessed_data = self.data.drop(columns=columns_to_drop)
            return preprocessed_data
        except:
            print("Columns already deleted")
            
    def check_missing_values(self,data):
        """
        Check for missing values in the DataFrame.
        
        Returns:
            DataFrame: A DataFrame showing the number of missing values per column.
        """
        self.data = data
        # Check for missing values using the isnull() method
        missing_values = self.data.isnull().sum()
        
        # Return the number of missing values per column
        return missing_values
    
    def fill_missing_values_auto(self, column_name,data):
        """
        Fill missing values in a specified column of a DataFrame using either the mean or median based on the data distribution.
        
        Args:
            column_name (str): The name of the column to fill missing values.
        
        Returns:
            DataFrame: A DataFrame with missing values filled in the specified column.
        """
        self.data = data
        # Calculate skewness
        skewness = self.data[column_name].skew()

        # Determine whether to use the mean or median based on the skewness
        if np.abs(skewness) < 0.5:
            fill_value = self.data[column_name].mean()
        else:
            fill_value = self.data[column_name].median()

        # Fill missing values with the calculated fill_value
        self.data[column_name].fillna(fill_value, inplace=True)

        return self.data
    
    def process_data(self):
        """
        Preprocess the data by dropping columns with the same value for all rows or all NaNs, and filling missing values in the "Attendance" and "Dist" columns with either the mean or median.

        Returns:
            DataFrame: The preprocessed DataFrame with relevant columns dropped and missing values filled.
        """
        preprocessed_data = self.drop_columns_with_same_values_or_all_nans()
        filled_data = self.fill_missing_values_auto("Attendance", preprocessed_data)
        filled_data = self.fill_missing_values_auto("Dist", filled_data)
        return filled_data


            


# In[24]:


if __name__ == "__main__":
    data = pd.read_csv("matches.csv")
    preprocessor = DataPreprocessing(data)
    data = preprocessor.process_data()
    print(preprocessor.check_missing_values(data))


# In[ ]:




