from index import KnowledgeBase
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.llamacpp import LlamaCpp


class Chatbot():
    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=2048,
            f16_kv=True,
            streaming=True
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            knowledge.get_index().as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )

    def invoke(self, chat_history):
        chat_history_tuples = []

        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))

        return self.chain.invoke({"question": chat_history[-1][0], "chat_history": chat_history_tuples})
