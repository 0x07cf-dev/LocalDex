from index import KnowledgeBase
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.llamacpp import LlamaCpp


class Chatbot():
    def __init__(self, knowledge: KnowledgeBase, type="llama", model="llama-2-7b-chat.Q4_K_M", **model_kwargs) -> None:
        self.vector_store = knowledge.get_index()
        self.retriever = knowledge.get_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
        self.__load_llm(type, model, **model_kwargs)
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.retriever
        )

    def __load_llm(self, type, model, format="gguf", **model_kwargs):
        default_path = f"models/{model}.{format}"

        match type.lower():
            case "llama":
                llm_kwargs = {
                    "model_path": default_path,
                    "n_gpu_layers": -1,
                    "n_ctx": 2048,
                    "f16_kv": True,
                    "streaming": True,
                    **model_kwargs
                }
                self.llm = LlamaCpp(**llm_kwargs)
            case _:
                raise ValueError

    def invoke(self, chat_history):
        chat_history_tuples = [(message[0], message[1]) for message in chat_history]
        return self.chain.invoke({"question": chat_history[-1][0], "chat_history": chat_history_tuples})
