from loguru import logger
from datetime import datetime

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOllama
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

class ChatMoodle:
    def __init__(self, token, llm_provider, model_name, model_cache, max_tokens, doc_language, courseid, instruction, history):

        self.token = token

        print("Using LLM-provider: " + llm_provider)
        self.llm_provider = llm_provider

        print("Using Model: " + model_name)
        self.model_name = model_name

        print("Model Cache directory: " + model_cache)
        self.model_cache = model_cache

        print("Max Prompt Tokens: " + str(max_tokens))
        self.max_tokens = max_tokens

        print("Using Language: " + doc_language)
        self.doc_language = doc_language

        print("On Course ID: " + str(courseid))
        self.courseid = str(courseid)

        # Initialize chat model.
        if llm_provider=="openai":
            self.chat = ChatOpenAI(temperature=0, model_name=model_name)
        else:
            raise ValueError("LLM-provider not recognized. Check LLM_PROVIDER environment variable.")

        print("Using local FAISS.")
        self.vector_store_dir = "vector_stores/course_" + self.courseid
        self.vector_store = FAISS.load_local(self.vector_store_dir,
        HuggingFaceInstructEmbeddings(cache_folder=self.model_cache,
        model_name="sentence-transformers/all-MiniLM-L6-v2"))

        self.history = history

        print("Using instruction: " + instruction)
        self.instruction = instruction


    def provide_context_for_question(self, query, smart_search=False):
        if smart_search==True:
            system="""
            You are an AI that provides assistance in database search.
            Please translate the user's query to a list of search keywords
            that will be helpful in retrieving documents from a database
            based on similarity.
            The language of the keywords should match the language of the documents:
            """+doc_language+"""\n
            Answer with a list of keywords.
            """
            query=self.chat(
                [SystemMessage(content=system),
                 HumanMessage(content=query)]
            ).content
        docs = self.vector_store.similarity_search(query)
        context = "\n---\n".join(doc.page_content for doc in docs)
        return context

    # Define functions for memory management
    def purge_memory(self, messages):
        token_count = self.token_counter(messages)
        if (len(messages)>1):
            while (token_count > int(os.getenv("MAX_PROMPT_TOKENS"))):
                print(token_count)
                # Print purged message for testing purposes
                # print("Purged the following message:\n" + messages[1])
                messages.pop(1)
                token_count = self.token_counter(messages)
        return token_count

    # PROMPT TOKEN COUNT DOES NOT EXACTLY MATCH OPENAI COUNT
    def token_counter(self, messages):
        # print("Counting tokens based on: " + current_model)
        if self.model_name == "gpt-4":
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        concatenated_content = ''.join([message.content for message in messages])
        token_count = len(encoding.encode(concatenated_content))
        return token_count

    def call_chat(self, query):

        # Search vector store for relevant documents
        context = self.provide_context_for_question(query)

        # Combine instructions + context to create system instruction for the chat model
        system_instruction = self.instruction + context

        # Convert message history to list of message objects
        print("History: " + str(self.history))
        messages_history = []
        i = 0
        for message in self.history:
            if i % 2 == 0:
                messages_history.append(HumanMessage(content=message))
            else:
                messages_history.append(AIMessage(content=message))
            i += 1

        print("Messages history: " + str(messages_history))
        # Initialize message list
        messages = [SystemMessage(content=system_instruction)]
        for message in messages_history:
            messages.append(message)
        messages.append(HumanMessage(content=query))

        # Purge memory to save tokens
        # Current implementation is not ideal.
        # Gradio keeps the entire history in memory
        # Therefore, the messages memory is re-purged on every call once token count max_tokens
        # print("Message purge")
        token_count = self.purge_memory(messages)
        # print("First message: \n" + str(messages[1].type))
        # print(str(messages))
        # print(token_count)
        if self.llm_provider != 'null':
            results = self.chat(messages)
            result_tokens = self.token_counter([results])
            print(f"Prompt tokens: {token_count}")
            print(f"Completion tokens: {result_tokens}")
            total_tokens = token_count+result_tokens
            print(f"Total tokens: {total_tokens}")
            results_content = results.content
        else:
            # debug mode:
            results_content = context

        return results_content