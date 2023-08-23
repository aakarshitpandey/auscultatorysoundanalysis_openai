# Description: This file contains the training code for the model
from os import path, getcwd

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.document_loaders import DataFrameLoader
from langchain.llms import OpenAI
from langchain.tools import tool, Tool
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from pandas import DataFrame
from numpy import array2string

from audioprocessor import AudioProcessor

@tool
def tool_get_mel_spectogram(file_path: str):
    """returns a mel spectogram for the given full file path"""
    spectogram = AudioProcessor.get_mel_spectogram(file_path)
    return array2string(spectogram, separator=",")

@tool
def tool_get_audio_file_path(file_path: str):
    """returns the full file path for the given relative path"""
    return AudioProcessor.get_audio_file_path(file_path)

prefix_prompt = """
You are an agent which helps the user to analyze the auscultatory (respiratory) breathing sounds of a patient.
You are given a relative path to an audio file. You have to use it to generate a mel spectogram of the patient's breathing sounds and you have to predict whether the patient has COVID-19 or not.
'COVID_Test_Status' of 0.0 means they do not have it. Value of 1.0 means the have COVID-19. You can use the following tools and data to help you with your task:
1) tool_get_audio_file_path - gets the full file path
2) tool_get_mel_spectogram - gets the mel spectogram for the given full file path
3) df - the dataframe containing the mel spectogram training data and the corresponding COVID_Test_Status"""

suffix_prompt = """
{chat_history}
Question: {user_input}
"""

class AuscultatorySoundAnalysisAgent:

    def __init__(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.agent_executor = self.create_agent()

    # Function to use train langchain model on the dataset returned by the data processor process_data function
    def create_agent(self) -> AgentExecutor:
        print("Creating agent...")
        db = self.create_embeddings()
        llm = OpenAI(temperature=0.7)
        prompt = ZeroShotAgent.create_prompt(
            prefix=prefix_prompt,
            suffix=suffix_prompt,
            input_variables=["user_input", "chat_history"]
        )
        
        print("Creating the conversational retrieval chain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            prompt=prompt,
        )

        # tools for helping the agent generate the mel spectogram
        tools = [
            Tool(
                name="Get MelSpectogram from a full file path",
                func=tool_get_mel_spectogram,
                description="useful for when you need to get a mel_spectogram for an audio file"
            ),
            Tool(
                name="Get full file path from a relative file path",
                func=tool_get_audio_file_path,
                description="useful for when you need to get the full file path for an audio file"
            )]
        
        agent = ZeroShotAgent(llm_chain=chain, tools=tools, verbose=True, name="Auscultatory Sound Analysis Agent")
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory(memory_key="chat_history"))

    # Create embeddings for the dataset and cache them using the FAISS vectorstore
    def create_embeddings(self) -> FAISS:
        print("Creating embeddings...")
        underlying_embeddings = OpenAIEmbeddings()

        # Instantiate a cache backed embeddings object
        fs = LocalFileStore(path.join(getcwd(), "__vector_cache__"))
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            fs,
            namespace=underlying_embeddings.model
        )

        # Load the dataframe into the vectorstore after splitting into chunks and store it into a FAISS vectorstore
        raw_data = DataFrameLoader(self.data_frame, page_content_column="Mel_Spectogram-Covid_Test_Status").load()
        
        if (raw_data == None) or (len(raw_data) == 0):
            raise Exception("No data found. Cannot create embeddings for an empty dataframe")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        documents = splitter.create_documents(raw_data)
        
        if documents.is_empty():
            raise Exception("Received empty documents. Cannot instantiate the FAISS vectorstore with empty documents")
        
        db = FAISS.from_documents(documents, cached_embedder)
        print("Embeddings created")
        return db

    def run(self, query: str):
        return self.agent_executor.run(query)