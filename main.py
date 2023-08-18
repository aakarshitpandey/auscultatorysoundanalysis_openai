import os
import pandas as pd
import constants

from dataprocessor import DataProcessor
from audioprocessor import AudioProcessor

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def main():
    dp = DataProcessor("data.csv")
    df = dp.process_data()
    print(df.groupby("COVID_test_status").count())
    idx = [81, 0]
    for index in idx:
        file_path = os.path.join(os.getcwd(), "dataset", (df["breathing-deep"][index]).lstrip("/"))
        print (df["COVID_test_status"][index])
        if os.path.isfile(file_path):
            AudioProcessor.show_mel_spectogram(file_path)

main()