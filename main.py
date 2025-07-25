import networkx as nx
from pyvis.network import Network
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
from datetime import datetime

app = FastAPI()

# --- Configuration ---
# Using Gemma 3n instruct model for improved extraction
MODEL_NAME = "google/gemma-3n-E2B-it" 

# --- Model Loading ---
# Note: This will download the model on first run, which may take some time.
# Ensure you have enough disk space.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# To make it runnable on most systems, we'll load in 4-bit for lower memory usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32
    # attn_implementation="eager" # Required for some systems
)

# --- Templating and Static Files ---
if not os.path.exists("graphs"):
    os.makedirs("graphs")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")
templates = Jinja2Templates(directory="templates")

# --- In-memory Graph Storage ---
G = nx.DiGraph()

def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """
    Uses the Gemma model to extract knowledge triples from the given text.
    """
    prompt = f"""
    Extract only explicit knowledge triples (subject, predicate, object) from the following text.
    Output ONLY the triples, one per line, in the format:
    subject, predicate, object

    Text: "{text}"
    Triples:
    """
    
    # Let the model's device_map handle tensor placement
    input_ids = tokenizer(prompt, return_tensors="pt")
    # Move inputs to the same device as the model
    input_ids = input_ids.to("cpu")
    
    outputs = model.generate(**input_ids, max_new_tokens=150)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the part of the response that contains the triples
    triples_text = response_text.split("Triples:")[1].strip()
    
    triples = []
    for line in triples_text.split("\n"):
        # Only accept lines with exactly two commas (i.e., three parts)
        if line.count(",") == 2:
            parts = [p.strip().strip('\"') for p in line.split(",")]
            if all(parts) and len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def create_graph_visualization(graph: nx.DiGraph, file_path: str):
    """
    Creates an interactive HTML visualization of the graph.
    """
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=True)
    net.from_nx(graph)
    
    # Custom physics options for a more stable layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      }
    }
    """)
    net.save_graph(file_path)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main page with the form.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process_text(request: Request, text: str = Form(...)):
    """
    Processes the text input, updates the graph, and returns the new graph visualization.
    """
    # 1. Extract triples from the text
    triples = extract_triples(text)
    
    # 2. Add triples to the graph
    for subj, pred, obj in triples:
        G.add_edge(subj, obj, label=pred)
        
    # 3. Create a new visualization
    graph_file = "graphs/knowledge_graph.html"
    create_graph_visualization(G, graph_file)
    
    # 4. Return the path to the visualization to be loaded in an iframe
    cache_buster = datetime.now().timestamp()
    return f'<iframe src="{graph_file}?v={cache_buster}" width="100%" height="750px"></iframe>' 