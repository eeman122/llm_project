# main.py

import gradio as gr

def greet(name):
    return f"Hello, {name}!"

app = gr.Interface(fn=greet, inputs="text", outputs="text")

app.launch(server_name="0.0.0.0", server_port=10000)
