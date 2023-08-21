import os
import pandas as pd

from audioprocessor import AudioProcessor

class DataProcessor:

    def __init__ (self, data_file_name: str):
        self.file_name = os.path.join(os.getcwd(), "dataset", data_file_name)

    # Clean this data and map covid status to 0 or 1 based on whether the person actually has covid or not
    def clean_data(self, df: pd.DataFrame):
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

        return df

    # Read the data from the csv file and clean it
    def process_data(self):
        print("Processing data...")
        df = pd.read_csv(self.file_name, encoding="ISO-8859-1")
        df = self.clean_data(df)
        return self.GenerateMelSpectogramForDataSet(df)
    
    # Generate Mel Spectogram for the entire dataset and return a new dataframe with the mel spectogram array and the covid status
    def GenerateMelSpectogramForDataSet(df: pd.DataFrame):
        
        # Add a column to the dataframe to store the mel spectogram while iterating through the dataframe
        for index, row in df.iterrows():
            try: 
                file_path = AudioProcessor.get_audio_file_path(row["breathing-deep"])
                spectogram = AudioProcessor.get_mel_spectogram(file_path)
                df.at[index, "mel_spectogram"] = spectogram
            except:
                df.drop(index, inplace=True)

        # Drop all columns except for the mel spectogram and the covid status. This df will be fed to the model
        df.drop(columns=[column for column in df.columns if column not in ["COVID_test_status", "mel_spectogram"]], inplace=True)

        return df
