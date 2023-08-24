# Description: This file contains the training code for the model
from os import path, getcwd

from langchain.agents import AgentExecutor, ZeroShotAgent, initialize_agent, AgentType
from langchain.document_loaders import DataFrameLoader
from langchain.llms import OpenAI
from langchain.tools import tool, Tool
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from pandas import DataFrame
from numpy import array2string, Inf as inf

from audioprocessor import AudioProcessor

@tool
def tool_get_mel_spectogram(file_path: str):
    """returns a mel spectogram for the given full file path"""
    spectogram = AudioProcessor.get_mel_spectogram(file_path)
    return "mel_spectogram: " + array2string(spectogram, separator=",")

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
3) df - the dataframe containing the mel spectogram training data and the corresponding COVID_Test_Status
4) A conversationalqachain built on the embeddings of a dataset which has the covid test status mapping to a melspectogram.\n
The user will most likely give you a relative file path. If the path looks like "/data/test/20200803/J5QGEKLK87d2dASH0umWBDzJ12P2/breathing-deep.wav" it is a relative path. If it is starting with D:/ or C:/ it is a full file path. You can use the tool_get_audio_file_path tool to get the full file path from the relative file path.
Context: {context}\n"""

suffix_prompt = """Begin!"
{chat_history}
Question: {question}
"""

class AuscultatorySoundAnalysisAgent:

    def __init__(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.agent_executor = self.create_agent()

    # Function to use train langchain model on the dataset returned by the data processor process_data function
    def create_agent(self) -> AgentExecutor:
        print("Creating agent...")
        db = self.create_embeddings()
        llm = OpenAI(temperature=0.5)

        # tools for helping the agent generate the mel spectogram
        tools = [
            Tool(
                name="Get MelSpectogram from a full file path",
                func=tool_get_mel_spectogram,
                description="useful for when you need to get a mel_spectogram for an audio file"
            ),
            Tool(
                name="Get full system file path from a relative file path",
                func=tool_get_audio_file_path,
                description="useful for when you need to get the full file path from a relative file path for an audio file"
            )
        ]

        prompt = PromptTemplate.from_template(input_variable=["context", "chat_history", "question"], template=prefix_prompt + suffix_prompt)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        print("Creating the conversational retrieval chain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            verbose=True,
            retriever=db.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs=dict(prompt=prompt)
        )

        chatbot_llm_chain= self.create_chatbot_chain_with_memory(llm=llm, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))

        tools.append(
            Tool(
                name="Covid test status from melspectogram",
                func=chain.run,
                description="useful for when you need to get the covid test status from a mel spectogram"
            )
        )

        tools.append(
            Tool(
                name="Chatbot",
                func=chatbot_llm_chain.run,
                description="useful for answering conversational questions from the user which do not involve getting the covid test status from a mel spectogram"
            )
        )
    
        return initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    def create_chatbot_chain_with_memory(self, llm: OpenAI, memory: ConversationBufferMemory) -> LLMChain:
        print("Creating chatbot chain...")
        prompt_text = """
            You are nice chatbot having a conversation with a human. You are given a question. You have to answer it.
            In addition to answering all the regular questions, you specialize in analyzing the auscultatory (respiratory) breathing sounds of a patient and predicting whether the patient has COVID-19 or not.
            For that you have access to a set of tools and data which are provided to you.
        """
        prompt = ChatPromptTemplate(messages=[
                    SystemMessagePromptTemplate.from_template(prompt_text),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}")
                ]
            )
        conversation = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
        return conversation


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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        documents = DataFrameLoader(data_frame=self.data_frame, page_content_column="COVID_test_status_mel_spectogram").load_and_split(splitter)
        
        if (documents == None) or (len(documents) == 0):
            raise Exception("No data found. Cannot create embeddings for an empty dataframe")
        
        db = FAISS.from_documents(documents, cached_embedder)
        print("Embeddings created")
        return db

    def run(self, query: str):
        return self.agent_executor.run(query)