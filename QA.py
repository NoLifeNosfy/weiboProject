import os
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class QA:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "None"
        os.environ["OPENAI_API_BASE"] = "http://172.29.7.155:8000/v1"
        self.embedding = HuggingFaceEmbeddings()
        self.db = Milvus(embedding_function=self.embedding, collection_name="arXiv_prompt",
                    connection_args={"host": "172.29.4.47", "port": "19530"})

        dbRetriever = self.db.as_retriever()


        self.llm = ChatOpenAI(temperature=0.0, model_name="vicuna-13b-v1.5")
        self.memory = ConversationBufferMemory()

        self.qa_stuff = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=dbRetriever,
            verbose=True,
            memory=self.memory
        )

        prompt = ChatPromptTemplate.from_template("{input}")
        self.conversation = LLMChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            prompt=prompt
        )

    def chat(self, query):
        return self.conversation.predict(input=query)

    def get_response(self, query):
        correctingMessage="Given a raw text input, determine whether the input contains wrong words. If so, output the corrected text. Else output the original text. Don't output any other words." \
                      "" \
                      "\n<< input >>\n" \
                      +query+"\n" \
                      "<< output >>"
        query = self.conversation.run(input=correctingMessage)
        print(query)


        selectMessage="Given a raw text input, determine whether the input is an academical question. If it is, output 'yes', else output 'no'. Don't take previous conversations into consideration. Don't output any other words."  \
                      "" \
                      "\n<< input >>\n" \
                      +query+"\n" \
                      "<< output >>"
        selected = self.conversation.run(input=selectMessage)

        print(selected)
        if(str(selected)=='no'):
            response = self.conversation.run(input=query)+"\nIt's not an academical question, so we have no arxiv reference"
            return response

        keyWord =  self.conversation.predict(input="Show me the key word of the following sentence. Don't say any other words.\n\n"+query)
        if keyWord[-1] == "." or keyWord[-1] == "," or keyWord[-1] == "\n":
            keyWord = keyWord[:-1]
        response = self.qa_stuff.run(query)

        if("I'm sorry" in str(response) or "i'm sorry" in str(response)):
            return response


        result = self.db.search(response, "similarity")
        metadata = result[0].metadata
        reference = '\r\n\r\n'+ "For a more detailed understanding of "+keyWord+', "'+str(metadata.get('title')).replace('\n','')+\
            '" by ' + str(metadata.get('authors')).replace('\n','') + ' is a useful resource, available at <https://arxiv.org/abs/' +\
            str(metadata.get('access_id'))+'>.'
        return response +reference

    def clear(self):
        self.memory.clear()


