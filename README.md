# PData: Personal Graph Database

PData is a project to create a personal, multi-modal graph database using language models. The goal is to build an application that can run on a mobile device, allowing users to build their own knowledge graph from various inputs like text, images, audio, and video.

## Project Vision

The core idea is to empower everyone to have their own personal knowledge base that is explorable and visual. Imagine taking a picture of a meal, and having it automatically connected to your nutrition goals, recipes you know, and restaurants you've visited. Or, recording a meeting and getting a graph of the key decisions and action items.

## Architecture

We will build this project in phases.

### Phase 1: Text-to-Graph Web PoC (Current)

This is our starting point. We'll build a web-based application that can:

1.  Take text input from a user.
2.  Use a `Gemma` model to extract knowledge triples (e.g., `(Subject, Predicate, Object)`).
3.  Store these triples in an in-memory graph database.
4.  Visualize the graph in an interactive way in the browser.

**Tech Stack:**

-   **Language Model:** A small, efficient `Gemma` model from Hugging Face.
-   **Backend:** Python with `FastAPI`.
-   **Graph Processing:** `networkx` for graph data structures.
-   **Graph Visualization:** `pyvis` to generate interactive HTML/JS visualizations.

### Phase 2: Multi-modal Inputs

In the next phase, we will add support for more input types:

-   **Images:** Use a vision-language model to describe images, then feed the description to our text-to-graph pipeline.
-   **Audio:** Use a speech-to-text model (like `Whisper`) to transcribe audio, then process the text.
-   **Video:** A combination of audio transcription and keyframe analysis from the video stream.

### Phase 3: Mobile Deployment

This is the most ambitious phase. We will need to:

1.  **Optimize the models:** Convert the models to a mobile-friendly format like TensorFlow Lite or ONNX Mobile.
2.  **Choose a mobile framework:** Options include React Native, Flutter, or native iOS/Android.
3.  **Develop the mobile app:** Create the UI/UX for capturing inputs and visualizing the graph on a small screen.
4.  **On-device database:** Select a lightweight, on-device database for storing the graph data persistently.

## How to Run (Phase 1)

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```

3.  **Open your browser:**
    Navigate to `http://127.0.0.1:8000` to see the application. 

## Privacy

- All data and models are stored and run locally on your device.
- No data is sent to the cloud or any third-party service.
- For extra security, restrict directory permissions (e.g., `chmod 700 graphs static templates`).
- For mobile deployment, use on-device models and encrypted local storage if needed.

## Mobile Deployment Notes

- Convert the Gemma model to TensorFlow Lite, ONNX, or CoreML for mobile use.
- Use Flutter or native Android/iOS for the app. See the `mobile/` directory for a suggested structure.
- All processing and storage remain on-device for privacy. 