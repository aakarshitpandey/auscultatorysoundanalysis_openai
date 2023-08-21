import os
import pandas as pd
import constants

from dataprocessor import DataProcessor
from audioprocessor import AudioProcessor
from agent import AuscultatorySoundAnalysisAgent

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def main():
    dp = DataProcessor("data.csv")
    df = dp.process_data()
    agent = AuscultatorySoundAnalysisAgent(df)
    
    while True:
        user_input = input("Type a question (or 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Thanks for talking.")
            break
    
        print(agent.run(user_input))

main()