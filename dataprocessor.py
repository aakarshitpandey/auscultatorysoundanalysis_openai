import os
import pandas as pd

from audioprocessor import AudioProcessor

class DataProcessor:

    def __init__ (self, data_file_name: str):
        self.file_name = os.path.join(os.getcwd(), "dataset", data_file_name)

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

        # for i in range(len(df)):
        #     try:
        #         file_path = AudioProcessor.get_audio_file_path(df["breathing-shallow"][i])
        #         y, sr = AudioProcessor.open_file(file_path)
        #         print(len(y)/sr)
        #     except:
        #         print("Error with " + (df["breathing-shallow"][i]).__str__())

        return df

    def process_data(self):
        print("Processing data...")
        df = pd.read_csv(self.file_name, encoding="ISO-8859-1")
        return self.clean_data(df)