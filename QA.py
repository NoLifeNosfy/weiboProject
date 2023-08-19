import os
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class QA:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "None"
        os.environ["OPENAI_API_BASE"] = "http://172.29.7.155:8000/v1"
        self.embedding = HuggingFaceEmbeddings()
        self.db = Milvus(embedding_function=self.embedding, collection_name="arXiv_prompt",
                    connection_args={"host": "172.29.4.47", "port": "19530"})

        dbRetriever = self.db.as_retriever()

        self.llm = ChatOpenAI(temperature=0.0, model_name="vicuna-13b-v1.5")

        self.qa_stuff = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=dbRetriever,
            verbose=True
        )

    def chat(self, query):
        return self.llm.predict(query)

    def get_response(self, query):
        keyWord =  self.llm.predict("Show me the key word of the following sentence. Don't say any other words.\n\n"+query)
        if keyWord[-1] == "." or keyWord[-1] == "," or keyWord[-1] == "\n":
            keyWord = keyWord[:-1]
        response = self.qa_stuff.run(query)

        if("I'm sorry" in str(response) or "i'm sorry" in str(response)):
            return response


        result = self.db.search(response, "similarity")
        metadata = result[0].metadata
        return response +'\r\n\r\n'+ "For a more detailed understanding of "+keyWord+', "'+str(metadata.get('title')).replace('\n','')+\
            '" by ' + str(metadata.get('authors')).replace('\n','') + ' is a useful resource, available at <https://arxiv.org/abs/' +\
            str(metadata.get('access_id'))+'>.'



