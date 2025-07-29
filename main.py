import networkx as nx
from pyvis.network import Network
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
import torch
import os
import gc
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from datetime import datetime
from PIL import Image
import io
import base64

app = FastAPI()

# --- Configuration ---
# Using Gemma 3n instruct model for improved extraction
# Change from text-only to vision model
MODEL_NAME = "google/gemma-3n-E2B-it" 

# Hugging Face token for accessing gated models
# Hugging Face token for accessing gated models
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this environment variable with your HF token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
# Note: This will download the model on first run, which may take some time.
# Ensure you have enough disk space.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# To make it runnable on most systems, we'll load in 4-bit for lower memory usage
# model = AutoModelForCausalLM.from_pretrained(
    # MODEL_NAME,
    # device_map=device,
    # torch_dtype=torch.float32
    # attn_implementation="eager" # Required for some systems
# )
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, torch_dtype="auto", token=HF_TOKEN).to(device)

def generate(messages):

    print(f"detailed messages: {messages}")
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device, dtype=model.dtype)
        
        print(f"Inputs created successfully")
        outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
        text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        
        print(f"Generated text: {text}")
        
        # clean-up the variables to free-up GPU RAM
        del inputs
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return text
    except Exception as e:
        print(f"Message Generation failed: {e}")

# --- Templating and Static Files ---
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Remove this line if you don't need it
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Keep this one - it's essential
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")
templates = Jinja2Templates(directory="templates")

# --- In-memory Graph Storage ---
G = nx.DiGraph()

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
async def process_multimodal(
    request: Request,
    text: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    """
    Processes text, image, and/or audio input using Gemma 3n model.
    """
    # Validate inputs and determine processing mode
    has_text = bool(text and text.strip())
    has_image = bool(image and image.filename and image.filename.strip())
    has_audio = bool(audio and audio.filename and audio.filename.strip())
    
    # Check if any input is provided
    if not any([has_text, has_image, has_audio]):
        return HTMLResponse(content="<p style='color: red;'>Please provide text, image, or audio input.</p>", status_code=400)
    
    # Validate file types
    if has_image and not image.content_type.startswith('image/'):
        return HTMLResponse(content="<p style='color: red;'>Error: Uploaded file is not a valid image.</p>", status_code=400)
    
    if has_audio and not audio.content_type.startswith('audio/'):
        return HTMLResponse(content="<p style='color: red;'>Error: Uploaded file is not a valid audio file.</p>", status_code=400)
    
    try:
        # Process files and extract triples
        triples = await process_inputs(text, image, audio, has_text, has_image, has_audio)
        
        # Add triples to the graph
        for subj, pred, obj in triples:
            G.add_edge(subj, obj, label=pred)

        # Create a new visualization
        graph_file = "graphs/knowledge_graph.html"
        create_graph_visualization(G, graph_file)

        return f'<iframe src="{graph_file}" width="100%" height="750px"></iframe>'
        
    except Exception as e:
        return HTMLResponse(content=f"<p style='color: red;'>Error getting triples and graph: {str(e)}</p>", status_code=400)


async def process_inputs(text: str, image: UploadFile, audio: UploadFile, 
                        has_text: bool, has_image: bool, has_audio: bool) -> list[tuple[str, str, str]]:
    """
    Unified function to process all input combinations and extract triples.
    """
    import tempfile
    import os
    
    # Prepare inputs for the unified function
    image_path = None
    audio_path = None
    
    if has_image:
        print("has image")
        # Save image to temporary file
        image_bytes = await image.read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_bytes)
            image_path = temp_file.name           
    
    if has_audio:
        # Save audio to temporary file
        print("has audio")
        audio_bytes = await audio.read()
        print(f"DEBUG: Audio file size: {len(audio_bytes)} bytes")
        print(f"DEBUG: Audio file name: {audio.filename}")
        print(f"DEBUG: Audio content type: {audio.content_type}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            audio_path = temp_file.name
            print(f"DEBUG: Audio saved to temporary file: {audio_path}")
            print(f"DEBUG: Temporary file exists: {os.path.exists(audio_path)}")
            print(f"DEBUG: Temporary file size: {os.path.getsize(audio_path)} bytes")
    
    try:
        print(f"DEBUG: Calling extract_triples_unified with:")
        print(f"DEBUG: - text: {text if has_text else None}")
        print(f"DEBUG: - image_path: {image_path}")
        print(f"DEBUG: - audio_path: {audio_path}")
        
        # Use the unified extraction function
        result = extract_triples_unified(
            text=text if has_text else None,
            image_path=image_path,
            audio_path=audio_path
        )
        print(f"DEBUG: extract_triples_unified returned: {result}")
        return result
    finally:
        # Clean up temporary files
        try:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
                print(f"DEBUG: Cleaned up image file: {image_path}")
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
                print(f"DEBUG: Cleaned up audio file: {audio_path}")
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {e}")

def extract_triples_unified(text: str = None, image_path: str = None, audio_path: str = None) -> list[tuple[str, str, str]]:
    """
    Unified function to extract knowledge triples from any combination of text, image, and audio.
    Uses Gemma 3n model for all processing.
    """
    try:
        # Build the prompt based on available inputs
        prompt_parts = []
        
        if text:
            prompt_parts.append(f'Text: "{text}"')
        
        if audio_path:
            print(f"DEBUG: Processing audio file: {audio_path}")
            # Process audio using Gemma 3n
            messages = [
              {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. Reply only with the answer to the question asked, and avoid using additional text in your response like 'here's the answer'."}]
        },
        {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Descripe the audio using Subject-verb-object sentences, avoiding duplicate descriptions. Because we need to get knowledge triples (subject, predicate, object) from the description."}
                ]
            }]
            try:
                audio_text = generate(messages)
                print(f" Audio processing result: {audio_text}")
                prompt_parts.append(f'Audio transcription: "{audio_text}"')
            except Exception as e:
                print(f"DEBUG: Audio processing failed: {e}")
                # prompt_parts.append('Audio transcription: "Audio processing failed"')
        
        if image_path:
            # Process image using Gemma 3n
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Descripe the image using Subject-verb-object sentences, avioding duplicate descriptions. Because we need to get knowledge triples (subject, predicate, object) from the description."}
                ]
            }]
            image_text = generate(messages)
            prompt_parts.append(f"Image description: {image_text}")
        
        # Create the combined prompt
        if len(prompt_parts) == 0:
            return []
        
        combined_prompt = f"""
        Extract knowledge triples (subject, predicate, object) from the following combined input.
        {chr(10).join(prompt_parts)}
        Provide the output as a list of comma-separated values, where each line is a new triple.
        Triples:
        """
        
        # Process with Gemma 3n model using the generate function
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": combined_prompt}
            ]
        }]
        
        response_text = generate(messages)
        
        # Extract triples
        triples_text = response_text.split("Triples:")[1].strip() if "Triples:" in response_text else response_text
        
        triples = []
        for line in triples_text.split("\n"):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    triples.append(tuple(parts))
        
        return triples
        
    except Exception as e:
        print(f"DEBUG: Unified extraction failed: {e}")
        return []
# def convert_audio_to_text(audio_bytes: bytes) -> str:
    # """
    # Converts audio bytes to text using speech recognition.
    # """
    # try:
        # import speech_recognition as sr
        # import tempfile
        # 
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # temp_file.write(audio_bytes)
            # temp_file_path = temp_file.name
        # 
        # recognizer = sr.Recognizer()
        # with sr.AudioFile(temp_file_path) as source:
            # audio_data = recognizer.record(source)
            # audio_text = recognizer.recognize_google(audio_data)
        # 
        # import os
        # os.unlink(temp_file_path)
        # return audio_text
        
    # except Exception as e:
        # print(f"Audio conversion failed: {e}")
        # return "Audio could not be transcribed"

 