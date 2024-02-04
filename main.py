from index import KnowledgeBase
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.llamacpp import LlamaCpp
import gradio as gr
import time

'''
    This is a small personal project to learn more about AI and Python.
    'Cause you don't need to be an astrophysicist to use a telescope:

    https://huggingface.co/docs
    https://www.gradio.app/docs
    https://python.langchain.com/docs


    I don't have the hardware needed to actually train or run models with many parameters.
    In fact, until AMD finally releases ROCm for Windows, I am forced to do inference with the CPU.

    Why did I switch to AMD last year? I guess I'll have to wait until 2026.
'''


if __name__ == "__main__":
    # Should probably parse args: input path, UI choice (cli-only or gradio)
    data_path = "data/"
    knowledge = KnowledgeBase(data_path)
    qa_chain = ConversationalRetrievalChain.from_llm(
        LlamaCpp(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=2048,
            f16_kv=True,
            streaming=True
        ),
        knowledge.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8, "k": 3}
        ),
        return_source_documents=True
    )  # TODO: move to chat.py

    def search(query, k):
        return knowledge.search(query, k)

    def bot(chat_history):
        chat_history_tuples = []
        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))

        response = qa_chain.invoke({"question": chat_history[-1][0], "chat_history": chat_history_tuples})
        answer = response["answer"]
        print(answer)

        chat_history[-1][1] = ""
        for c in answer:
            chat_history[-1][1] += c
            time.sleep(0.03)
            yield chat_history

    def user(query, chat_history):
        return "", chat_history + [[query, ""]]

    # Gradio UI
    with gr.Blocks() as demo:
        # TODO: a sidebar that displays source data... data_dir choice via gradio component
        with gr.Row():
            chatbot = gr.Chatbot(height=600)
            textbox_search_output = gr.Textbox(placeholder="", container=False, scale=1)

        with gr.Row():
            with gr.Column():
                textbox_chat = gr.Textbox(label="Ask Questions", placeholder="Waiting for your question...", container=True)
                clear_chat = gr.ClearButton(textbox_chat)

                textbox_chat.submit(user, [textbox_chat, chatbot], [textbox_chat, chatbot], queue=False).then(bot, chatbot, chatbot)

            with gr.Column():
                textbox_search = gr.Textbox(label="Search Knowledge", placeholder="Waiting for your query...", container=True)
                slider = gr.Slider(1, 8, value=4, step=1)

                textbox_search.submit(search, [textbox_search, slider], [textbox_search_output])

    demo.launch()
