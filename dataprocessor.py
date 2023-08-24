import os
import pandas as pd

from numpy import array2string, Inf as inf
from audioprocessor import AudioProcessor

class DataProcessor:

    def __init__ (self, data_file_name: str):
        self.file_name = os.path.join(os.getcwd(), "dataset", data_file_name)

    # Clean this data and map covid status to 0 or 1 based on whether the person actually has covid or not
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Cleaning data...")
        selected_columns = ["breathing-deep", "breathing-shallow", "COVID_STATUS", "COVID_test_status"]
        drop_columns = [column for column in df.columns if column not in selected_columns]
        df.drop(columns=drop_columns, inplace=True)

        # map the COVID_test_status column to whether the person actually has covid or not
        df["COVID_test_status"] = df["COVID_STATUS"].map({
            "healthy": 0.0,
            "resp_illness_not_identified": 0.0,
            "recovered_full": 0.0,
            "no_resp_illness_exposed": 1.0,
            "positive_mild": 1.0,
            "positive_moderate": 1.0,
            "positive_asymp": 1.0,
        })

        # Iterate through the dataset and drop a row with covid_test_status 0.0 after we have had 150 rows covid_test_status 0.0
        # This has been done to balance the dataset
        covid_negative_count = 0
        for index, row in df.iterrows():
            try:
                if (row["COVID_test_status"] == 0.0):
                    covid_negative_count += 1
                    if (covid_negative_count > 150):
                        df.drop(index, inplace=True)
            except Exception as e:
                df.drop(index, inplace=True)

        return df
    
    # Generate Mel Spectogram for the entire dataset and return a new dataframe with the mel spectogram array and the covid status
    def GenerateMelSpectogramForDataSet(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Generating mel spectogram for the dataset...", df.head())
        # Add a column to the dataframe to store the mel spectogram while iterating through the dataframe
        for index, row in df.iterrows():
            try: 
                file_path = AudioProcessor.get_audio_file_path(row["breathing-deep"])
                spectogram = AudioProcessor.get_mel_spectogram(file_path)
                serialized_spectogram = array2string(spectogram, separator=",")
                df.at[index, "mel_spectogram"] = serialized_spectogram
            except Exception as e:
                # Drop the row if the file path is invalid
                df.drop(index, inplace=True)

        # Drop all columns except for the mel spectogram and the covid status. This df will be fed to the model
        df.drop(columns=[column for column in df.columns if column not in ["COVID_test_status", "mel_spectogram"]], inplace=True)
        print(df.head())
        return df
    
    # Take df which has COVID_test_status and mel_spectogram columns and return a new dataframe with a single column serializing the two columns into one
    def UnifyColumnsForDataSet(self, df:pd.DataFrame) -> pd.DataFrame:
        print("Unifying columns for the dataset...")
        unified_column_name = "COVID_test_status_mel_spectogram"
        for index, row in df.iterrows():
            try:
                df.at[index, unified_column_name] = "COVID_test_status: " + str(row["COVID_test_status"]) + ", " + "mel_spectogram: " + row["mel_spectogram"]
            except Exception as e:
                # Drop the row if the file path is invalid
                print(e)
                df.drop(index, inplace=True)
        
        # Drop all columns except for the unified columns. This df will be fed to the model
        df.drop(columns=[column for column in df.columns if column not in [unified_column_name]], inplace=True)
        print(df.head())

        # Save the processed data to a csv file
        df.to_csv(os.path.join(os.getcwd(), "dataset", "processed_data.csv"), index=False, encoding="utf-8")

        return df
    
    # Read the data from the csv file and clean it
    def process_data(self) -> pd.DataFrame:
        print("Processing data...")

        # Check if the data processing has already been done
        processed_data_file_path = os.path.join(os.getcwd(), "dataset", "processed_data.csv")
        if (os.path.exists(processed_data_file_path)):
            return pd.read_csv(processed_data_file_path, encoding="utf-8")
        
        # Read the data from the csv file and clean it
        df = pd.read_csv(self.file_name, encoding="ISO-8859-1")
        df = self.clean_data(df)
        df = self.GenerateMelSpectogramForDataSet(df)
        return self.UnifyColumnsForDataSet(df)
