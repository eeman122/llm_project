# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Used Car Price Prediction API is running."}



def greet(name):
    return f"Hello, {name}!"

app = gr.Interface(fn=greet, inputs="text", outputs="text")

app.launch(server_name="0.0.0.0", server_port=10000)
