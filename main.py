# main.py
from fastapi import FastAPI
import gradio as gr
import threading

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, your app is running!"}

def greet(name):
    return f"Hello, {name}!"

def run_gradio():
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch(server_name="0.0.0.0", server_port=7860)  # Gradio port

if __name__ == "__main__":
    import uvicorn
    import threading

    # Run Gradio app in a separate thread
    thread = threading.Thread(target=run_gradio)
    thread.start()

    # Run FastAPI app on port 10000
    uvicorn.run(app, host="0.0.0.0", port=10000)
