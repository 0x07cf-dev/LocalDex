import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


'''
    Files are loaded from data_path and split into smaller chunks.
    All these chunks are converted to embeddings which are saved to a vector store (FAISS) locally at index_path.
        (!) This operation is sequential and doesn't scale well. My usecase is tiny.
'''


class KnowledgeBase():
    def __init__(self, data_path=".data/", index_path="faiss_index/", device="cpu") -> None:
        self.data_path = data_path
        self.index_path = index_path
        self.device = device
        self.document_loaders = {
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".pdf": PyPDFLoader,
        }
        self.__create_embeddings()

        try:
            self.vector_store = FAISS.load_local(index_path, self.embeddings)
        except RuntimeError:
            print("Index not found.")
            self.__make_index()

    def __create_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": False}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def __load_data(self, chunk_size=1000, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Create a list of documents from all .txt, .docx and .pdf files in data_path
        documents = []
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            _, file_ext = os.path.splitext(file)
            document = None

            if file_ext in self.document_loaders:
                loader = self.document_loaders[file_ext]
                document = loader(file_path).load()
                print(f"Loaded {file_path} as {file_ext}")

            if document is not None:
                documents.extend(document)

        return splitter.split_documents(documents)

    def __make_index(self):
        chunks = self.__load_data()
        print("This could take a few minutes...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.index_path)
        print("Knowledge index created.")

    def search(self, query, num_results=3):
        results = self.vector_store.similarity_search(query, k=num_results)
        retval = ""
        for i in range(num_results):
            chunk = results[i]
            source = chunk.metadata["source"]
            retval = retval + f"From: {source}\n\n"
            retval = retval + chunk.page_content
            retval = retval + "\n\n"
        return retval
