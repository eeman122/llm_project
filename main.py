import os
import uvicorn
from fastapi import FastAPI
import gradio as gr  # if you use Gradio

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, your app is running!"}

def greet(name):
    return f"Hello, {name}!"

# Option 1: Gradio separate server
def run_gradio():
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # If you want to run Gradio and FastAPI concurrently
    import threading
    threading.Thread(target=run_gradio).start()
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
