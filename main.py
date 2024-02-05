from chat import Chatbot
from index import KnowledgeBase
import gradio as gr
import time
import os

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

data_path = "data/"
index_path = "faiss_index/"
knowledge = None
chatbot = None


# Gradio UI
with gr.Blocks() as demo:
    gr.Theme(primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.amber)
    title = gr.Markdown("""
        # Where is your data?
        Click below to specify a directory.
    """)

    with gr.Column(visible=True) as col1:
        data_display = gr.FileExplorer(glob="**/", ignore_glob="[._~]*/", file_count="single")
        button_data_choose = gr.Button("Click to select directory")

    with gr.Row(visible=False) as row1:
        chatbot_box = gr.Chatbot(
            height=600,
            label="Chat with your Documents",
            show_label=True,
            show_copy_button=True,
            bubble_full_width=False,
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.jpg")))
        )
        textbox_search_results = gr.Textbox(placeholder="", label="Search Results", show_label=True, show_copy_button=True, lines=25, max_lines=25, autoscroll=False)

    with gr.Row(visible=False) as row2:
        with gr.Column():
            textbox_chat = gr.Textbox(label="Ask Questions", placeholder="Waiting for your question...", container=True)

            with gr.Row():
                clear_chat = gr.ClearButton(textbox_chat)
                button_upload_chat = gr.UploadButton("📁", file_types=["text"])

        with gr.Column():
            textbox_search = gr.Textbox(label="Search Knowledge", placeholder="Waiting for your query...", container=True)
            slider_search_k = gr.Slider(1, 8, value=4, step=1)

    def find_index():
        path_missing = not index_path or not os.path.exists(index_path) or not os.path.isdir(index_path)
        faiss_index, faiss_pickle = os.path.join(index_path, "index.faiss"), os.path.join(index_path, "index.pkl")

        if path_missing:
            if not os.path.exists(faiss_index) or not os.path.exists(faiss_pickle):
                print(f"Expected index not found: \n{faiss_index}\n{faiss_pickle}")
                return {
                    col1: gr.Column(visible=True),
                    row1: gr.Row(visible=False),
                    row2: gr.Row(visible=False)
                }

        print(f"Found index: \n{faiss_index}\n{faiss_pickle}")
        return {
            col1: gr.Column(visible=False),
            row1: gr.Row(visible=True),
            row2: gr.Row(visible=True)
        }

    def select_data_dir(path):
        print(f"Selected data directory: {path}")
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            return data_display, title, gr.Column(visible=True), gr.Row(visible=False), gr.Row(visible=False)

        global knowledge
        global chatbot
        knowledge = KnowledgeBase(path)
        chatbot = Chatbot(knowledge)
        return {
            data_display: path,
            col1: gr.Column(visible=False),
            row1: gr.Row(visible=True),
            row2: gr.Row(visible=True),
            title: gr.Markdown("""
                               # LocalDex
                               ### Chat with your documents.
                               """)
        }

    def search(query, k):
        return knowledge.search(query, k)

    def print_like_dislike(x: gr.LikeData):
        print(x.index, x.liked)

    def user_message(query, chat_history):
        return chat_history + [[query, ""]], gr.Textbox(value="...", interactive=False)

    def user_file(file, chat_history):
        return chat_history + [[file.name]], gr.Textbox(value="...", interactive=False)

    def bot_message(chat_history):
        response = chatbot.invoke(chat_history)
        for chr in response["answer"]:
            chat_history[-1][1] += chr

            if chr in ".!?":
                time.sleep(0.3)
            elif chr in ",:;":
                time.sleep(0.1)
            else:
                time.sleep(0.025)

            yield chat_history, gr.Textbox(value="", interactive=True)  # FIXME: no, ur too tired, sleep

    # Event handlers
    demo.load(find_index, None, [col1, row1, row2])
    button_data_choose.click(select_data_dir, data_display, [data_display, title, col1, row1, row2])

    file_msg = button_upload_chat.upload(user_file, [button_upload_chat, chatbot_box], [chatbot_box, textbox_chat], queue=False).then(
        fn=bot_message,
        inputs=chatbot_box,
        outputs=chatbot_box
    )
    textbox_chat.submit(fn=user_message, inputs=[textbox_chat, chatbot_box], outputs=[chatbot_box, textbox_chat], queue=False).then(
        fn=bot_message,
        inputs=chatbot_box,
        outputs=[chatbot_box, textbox_chat]
    )
    textbox_search.change(
        fn=search,
        inputs=[textbox_search, slider_search_k],
        outputs=[textbox_search_results]
    )
    chatbot_box.like(print_like_dislike, None, None)

if __name__ == "__main__":
    # Should probably parse args: input path, UI choice (cli-only or gradio)
    demo.queue()
    demo.launch()
