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
import cv2

# Load environment variables from .env file
load_dotenv()
import datetime
from PIL import Image
import io
import base64

app = FastAPI()

# --- Configuration ---
# Using Gemma 3n  model for multimodal processing
MODEL_NAME = "google/gemma-3n-E2B-it" 
# Hugging Face token for accessing gated models
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this environment variable with your HF token
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- Model Loading ---
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, torch_dtype="auto", token=HF_TOKEN).to(device)

#function to process all inputs with gemma 3n model
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

app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")
templates = Jinja2Templates(directory="templates")

# --- In-memory Graph Storage ---
G = nx.DiGraph()

# Initialize personal analytics
from personal_analytics import PersonalAnalytics
from personal_analytics_integration import integrate_analytics_to_main_app, set_analytics_instance

personal_analytics = PersonalAnalytics()
set_analytics_instance(personal_analytics)

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
    audio: UploadFile = File(None),
    video: UploadFile = File(None)
):
    """
    Processes text, image, and/or audio input using Gemma 3n model.
    """
    # Validate inputs and determine processing mode
    has_text = bool(text and text.strip())
    has_image = bool(image and image.filename and image.filename.strip())
    has_audio = bool(audio and audio.filename and audio.filename.strip())
    has_video = bool(video and video.filename and video.filename.strip())
    
    # Check if any input is provided
    if not any([has_text, has_image, has_audio, has_video]):
        return HTMLResponse(content="<p style='color: red;'>Please provide text, image, audio, or video input.</p>", status_code=400)
    
    # Validate file types
    if has_image and not image.content_type.startswith('image/'):
        return HTMLResponse(content="<p style='color: red;'>Error: Uploaded file is not a valid image.</p>", status_code=400)
    
    if has_audio and not audio.content_type.startswith('audio/'):
        return HTMLResponse(content="<p style='color: red;'>Error: Uploaded file is not a valid audio file.</p>", status_code=400)
    
    if has_video and not video.content_type.startswith('video/'):
        return HTMLResponse(content="<p style='color: red;'>Error: Uploaded file is not a valid video file.</p>", status_code=400)
    
    try:
        # Process files and extract triples
        triples = await process_inputs(text, image, audio, video, has_text, has_image, has_audio, has_video)
        
        # Add triples to the graph
        for subj, pred, obj in triples:
            G.add_edge(subj, obj, label=pred)

        # Load triples into personal analytics for categorization and reporting
        # Use current date as the activity date
        personal_analytics.load_graph_from_triples(triples, activity_date=datetime.datetime.now().date())

        # Create a new visualization
        graph_file = "graphs/knowledge_graph.html"
        create_graph_visualization(G, graph_file)

        # Return enhanced HTML response with analytics links
        return HTMLResponse(content=f"""
        <html>
        <head>
            <title>Knowledge Graph Visualization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .triples {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .graph-container {{ border: 1px solid #ddd; border-radius: 5px; }}
                .actions {{ margin: 20px 0; }}
                .btn {{ 
                    display: inline-block; 
                    padding: 10px 20px; 
                    margin: 5px; 
                    background-color: #007bff; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    transition: background-color 0.3s;
                }}
                .btn:hover {{ background-color: #0056b3; }}
                .btn-success {{ background-color: #28a745; }}
                .btn-success:hover {{ background-color: #1e7e34; }}
                .btn-info {{ background-color: #17a2b8; }}
                .btn-info:hover {{ background-color: #138496; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Knowledge Graph Visualization</h1>
                <p>Extracted {len(triples)} knowledge triples from your input.</p>
                <div class="actions">
                    <a href="/" class="btn">← Back to Input</a>
                    <a href="/analytics" class="btn btn-success">📊 Personal Analytics</a>
                    <a href="/analytics/healthcare" class="btn btn-info">🏥 Healthcare Report</a>
                    <a href="/analytics/education_work" class="btn btn-info">📚💼 Education & Work Report</a>
                </div>
            </div>
            
            <div class="triples">
                <h2>Extracted Triples:</h2>
                <ul>
                    {''.join(f'<li><strong>{subj}</strong> --{pred}--> <strong>{obj}</strong></li>' for subj, pred, obj in triples)}
                </ul>
            </div>
            
            <div class="graph-container">
                <iframe src="{graph_file}" width="100%" height="750px"></iframe>
            </div>
        </body>
        </html>
        """)
        
    except Exception as e:
        return HTMLResponse(content=f"<p style='color: red;'>Error getting triples and graph: {str(e)}</p>", status_code=400)


async def process_inputs(text: str, image: UploadFile, audio: UploadFile, video: UploadFile,
                        has_text: bool, has_image: bool, has_audio: bool, has_video: bool) -> list[tuple[str, str, str]]:
    """
    Unified function to process all input combinations and extract triples.
    """
    import tempfile
    import os
    
    # Prepare inputs for the unified function
    image_path = None
    audio_path = None
    video_path = None
    
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
    
    if has_video:
        # Save video to temporary file
        print("has video")
        video_bytes = await video.read()
        print(f"DEBUG: Video file size: {len(video_bytes)} bytes")
        print(f"DEBUG: Video file name: {video.filename}")
        print(f"DEBUG: Video content type: {video.content_type}")
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_bytes)
            video_path = temp_file.name
            print(f"DEBUG: Video saved to temporary file: {video_path}")
            print(f"DEBUG: Temporary file exists: {os.path.exists(video_path)}")
            print(f"DEBUG: Temporary file size: {os.path.getsize(video_path)} bytes")
    
    try:
        print(f"DEBUG: Calling extract_triples_unified with:")
        print(f"DEBUG: - text: {text if has_text else None}")
        print(f"DEBUG: - image_path: {image_path}")
        print(f"DEBUG: - audio_path: {audio_path}")
        print(f"DEBUG: - video_path: {video_path}")
        
        # Use the unified extraction function
        result = extract_triples_unified(
            text=text if has_text else None,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path
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
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
                print(f"DEBUG: Cleaned up video file: {video_path}")
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {e}")

def extract_triples_unified(text: str = None, image_path: str = None, audio_path: str = None, video_path: str = None) -> list[tuple[str, str, str]]:
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
        
        if image_path:
            # Process image using Gemma 3n
            image = resize_image(image_path)
            messages = [
              {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Reply only with the answer to the question asked, and avoid using additional text in your response like 'here's the answer'."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Descripe the image using Subject-verb-object sentences, avioding duplicate descriptions. Because we need to get knowledge triples (subject, predicate, object) from the description."}
                ]
            }]
            image_text = generate(messages)
            prompt_parts.append(f"Image description: {image_text}")
        
        if video_path:
            print(f"DEBUG: Processing video file: {video_path}")
            
            # Create frames directory if it doesn't exist
            frames_dir = "frames"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
                print(f"DEBUG: Created frames directory: {frames_dir}")
            
            # Extract frames from video
            video_frames = extract_frames(video_path, num_frames=10)
            
            if video_frames:
                # Process video using Gemma 3n with frames
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant. Reply only with the answer to the question asked, and avoid using additional text in your response like 'here's the answer'."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the video using Subject-verb-object sentences, avoiding duplicate descriptions. Because we need to get knowledge triples (subject, predicate, object) from the description."}
                        ]
                    }
                ]
                
                # Add frames to the messages structure
                for frame_data in video_frames:
                    print(f"DEBUG: Processing frame: {frame_data}")
                    img, timestamp = frame_data
                    frame_path = f"{frames_dir}/frame_{timestamp}.png"
                    img.save(frame_path)
                    print(f"DEBUG: Saved frame to: {frame_path}")
                    messages[1]["content"].append({"type": "image", "image": frame_path})
                
                try:
                    print(f"DEBUG: Video messages with frames: {messages}")
                    video_text = generate(messages)
                    print(f"Video processing result: {video_text}")
                    prompt_parts.append(f"Video description: {video_text}")
                except Exception as e:
                    print(f"DEBUG: Video processing failed: {e}")
                    
                # Clean up frame files
                for frame_data in video_frames:
                    img, timestamp = frame_data
                    frame_path = f"{frames_dir}/frame_{timestamp}.png"
                    if os.path.exists(frame_path):
                        os.unlink(frame_path)
                        print(f"DEBUG: Cleaned up frame: {frame_path}")
            else:
                print(f"DEBUG: No frames extracted from video")
                # direct video processing
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant. Reply only with the answer to the question asked, and avoid using additional text in your response like 'here's the answer'."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": video_path},
                                {"type": "text", "text": "Describe the video using Subject-verb-object sentences, avoiding duplicate descriptions. Because we need to get knowledge triples (subject, predicate, object) from the description."}
                            ]
                        }
                    ]
                    video_text = generate(messages)
                    print(f"Video processing result (direct): {video_text}")
                    prompt_parts.append(f"Video description: {video_text}")
                except Exception as e:
                    print(f"DEBUG: Direct video processing failed: {e}")
        
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
        
        # Extract triples from the response
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

def extract_frames(video_path, num_frames):
    """
    The function is adapted from:
    https://github.com/merveenoyan/smol-vision/blob/main/Gemma_3_for_Video_Understanding.ipynb
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the step size to evenly distribute frames across the video.
    step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = round(frame_idx / fps, 2)
        frames.append((img, timestamp))
    print(f"DEBUG: Extracted {len(frames)} frames from video")
    cap.release()
    return frames

def resize_image(image_path):
    img = Image.open(image_path)

    target_width, target_height = 640, 640
    # Calculate the target size (maximum width and height).
    if target_width and target_height:
        max_size = (target_width, target_height)
    elif target_width:
        max_size = (target_width, img.height)
    elif target_height:
        max_size = (img.width, target_height)

    img.thumbnail(max_size)

    return img

# Integrate analytics functionality
integrate_analytics_to_main_app(app)
