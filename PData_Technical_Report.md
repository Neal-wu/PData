# PData: A Unified Multimodal Personalized Graph Database System Powered by Gemma 3n

## Abstract

This paper presents PData, a novel personal knowledge graph system that leverages Google's Gemma 3n model for unified multimodal processing. Unlike traditional approaches that require separate preprocessing pipelines for different input modalities, PData demonstrates how a single, unified AI model can handle text, image, audio, and video inputs natively, enabling real-time personal analytics across life categories. We discuss the architectural decisions, technical challenges, and innovative use of Gemma 3n's unique capabilities to create a privacy-preserving, on-device personal graph database.

## 1. Introduction

Personal knowledge management systems have traditionally relied on complex multi-stage pipelines, requiring separate models for different input modalities. This approach introduces latency, complexity, and potential privacy concerns. PData addresses these limitations by leveraging Gemma 3n's native multimodal capabilities to create a unified, on-device personal analytics system.

### 1.1 Problem Statement

Traditional personal analytics systems face several challenges:
- **Modality Fragmentation**: Separate processing pipelines for text, image, audio, and video
- **Privacy Concerns**: Cloud-based processing of personal data
- **Complexity**: Multiple model dependencies and preprocessing steps
- **Latency**: Sequential processing across different modalities
- **Resource Overhead**: Multiple models consuming significant computational resources

### 1.2 Innovation

PData introduces a novel approach by utilizing Gemma 3n's unified multimodal architecture to process all input types through a single model, eliminating the need for separate preprocessing pipelines while maintaining privacy through on-device processing.

## 2. System Architecture

### 2.1 High-Level Architecture

PData employs a three-tier architecture designed for efficiency and privacy:

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Input     │  │ Analytics   │  │   Visualization     │  │
│  │  Interface  │  │ Dashboard   │  │     Dashboard       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Application Layer                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Request    │  │ Analytics   │  │   Graph             │  │
│  │  Handler    │  │ Engine      │  │  Generator          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Gemma 3n Processing Layer                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Unified Multimodal Processing                 │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────┐ │ │
│  │  │  Text   │ │  Image  │ │  Audio  │ │    Video      │ │ │
│  │  │Processing│ │Processing│ │Processing│ │  Processing   │ │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 Unified Input Processing
The system accepts four input modalities through a single interface:
- **Text**: Natural language descriptions of activities
- **Image**: Visual content (photos, screenshots, documents)
- **Audio**: Voice recordings and audio files
- **Video**: Video recordings with temporal content

#### 2.2.2 Analytics Engine
The analytics engine categorizes activities into five life areas:
- **Education & Work**: Learning activities, professional tasks
- **Healthcare**: Exercise, medication, wellness activities
- **Entertainment**: Media consumption, hobbies, social activities
- **Finance**: Spending, investments, financial management
- **Other**: Miscellaneous life activities

## 3. Technical Implementation

### 3.1 Gemma 3n Integration

```python
MODEL_NAME = "google/gemma-3n-E2B-it"
HF_TOKEN = 'your HF token'
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, 
    torch_dtype="auto", 
    token=HF_TOKEN
).to(device)

def extract_triples_unified(text=None, image_path=None, audio_path=None, video_path=None):
    messages = []
    
    if text:
        messages.append({"role": "user", "content": text})
    if image_path:
        image = Image.open(image_path)
        messages.append({"role": "user", "content": [{"type": "image", "image": image}]})
    if audio_path:
        messages.append({"role": "user", "content": [{"type": "audio", "audio": audio_path}]})
    if video_path:
        messages.append({"role": "user", "content": [{"type": "video", "video": video_path}]})
    
    # Single model call for all modalities
    response = generate(messages)
    return parse_triples(response)
```

**Key Technical Choices:**
- **Model Size**: resource_based parameters size for optimal on-device performance
- **Quantization**: Automatic dtype optimization for memory efficiency
- **Unified Processing**: Single model handles all input modalities

### 3.2 Knowledge Graph Generation

The system extracts structured knowledge triples using a specialized prompt:

```python
 prompt = f"""
 Extract knowledge triples (subject, predicate, object) from the following combined input.
 {chr(10).join(input_messages)}
 Provide the output as a list of comma-separated values, where each line is a new triple. 
 Triples:
 """
```

### 3.3 Analytics and Categorization

```python
def categorize_triple(self, triple: Tuple[str, str, str]) -> LifeCategory:
    text_to_check = f"{triple[0]} {triple[1]} {triple[2]}".lower()
    
    for category, keywords in self.category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_to_check:
                return category
    
    return LifeCategory.OTHER
```

## 4. Innovative Use of Gemma 3n Features

### 4.1 Native Multimodality

**Traditional Approach:**
```
Text → Text Model → Triples
Image → Vision Model → Text → Text Model → Triples
Audio → STT Model → Text → Text Model → Triples
Video → Frame Extraction → Vision Model → Text → Text Model → Triples
```

**PData's Unified Approach:**
```
Text/Image/Audio/Video → Gemma 3n → Triples
```

**Benefits:**
- **Reduced Latency**: Single model inference vs. multiple pipeline stages
- **Consistent Quality**: Same model ensures uniform extraction quality
- **Simplified Architecture**: No modality-specific preprocessing
- **Lower Resource Usage**: Single model vs. multiple specialized models

### 4.2 On-Device Performance

**Memory Optimization:**
- **Quantization**: Automatic dtype optimization reduces memory footprint
- **Model Size**: flexible model size choices to optimize resource usage

### 4.3 Privacy-Preserving Architecture

**On-Device Processing:**
- All AI processing occurs locally
- No data transmission to external services
- Secure local storage of personal information

**Data Flow:**
```
User Input → Local Gemma 3n → Local Storage → Local Analytics
```

## 5. Technical Challenges and Solutions

### 5.1 Challenge: Unified Input Handling

**Problem**: Different input modalities require different preprocessing approaches.

**Solution**: Leveraged Gemma 3n's native multimodal capabilities with unified message structure for all modalities.

### 5.2 Challenge: Memory Management

**Problem**: Large language models require significant memory resources.

**Solution**: Implemented efficient memory management with automatic cleanup after inference. Resource-based model size choice.

### 5.3 Challenge: Temporal Data Tracking

**Problem**: Need to track when activities occurred for meaningful analytics.

**Solution**: Implemented date-aware processing with activity timestamps.

### 5.4 Challenge: Real-time Analytics Generation

**Problem**: Need to generate comprehensive reports quickly.

**Solution**: Implemented efficient analytics engine with keyword-based categorization.

## 6. Why These Technical Choices Were Right

### 6.1 Unified Model Approach

**Why Gemma 3n was the right choice:**

1. **Native Multimodality**: Unlike other models that require separate encoders, Gemma 3n provides superior cross-modal understanding.

2. **Efficient Architecture**: The 2B parameter size provides excellent performance while maintaining reasonable resource requirements for on-device deployment.

3. **Privacy-First Design**: On-device processing eliminates privacy concerns associated with cloud-based solutions.

4. **Simplified Maintenance**: Single model reduces complexity and maintenance overhead.

### 6.2 FastAPI Backend

**Why FastAPI was the right choice:**

1. **Async Processing**: Handles multiple concurrent requests efficiently
2. **Automatic Documentation**: Built-in OpenAPI/Swagger documentation
3. **Type Safety**: Python type hints provide better code reliability
4. **Performance**: High-performance web framework suitable for real-time applications

### 6.3 Analytics Architecture

**Why the keyword-based categorization was the right choice:**

1. **Transparency**: Clear, interpretable categorization rules
2. **Customizability**: Easy to modify categories and keywords
3. **Performance**: Fast categorization without additional model inference
4. **Domain Adaptability**: Can be easily adapted for different life areas

## 7. Innovation Assessment

### 7.1 Novel Contributions

1. **Unified Multimodal Processing**: First system to use Gemma 3n's native multimodal capabilities for personal knowledge graph generation.

2. **On-Device Personal Analytics**: Complete privacy-preserving personal analytics system running entirely on-device.

3. **Real-time Categorization**: Immediate categorization and insights generation without batch processing.

4. **Temporal Knowledge Graphs**: Date-aware knowledge graph construction for meaningful temporal analysis.

### 7.2 Gemma 3n's Unique Features Utilization

**Mix'n'Match Capabilities:**
- Leveraged Gemma 3n's ability to process any combination of modalities in a single inference call
- Implemented dynamic message construction based on available inputs
- Enabled seamless switching between different input combinations

**On-Device Performance:**
- Optimized for local deployment with memory-efficient processing
- Implemented automatic cleanup to prevent memory leaks
- Designed for real-time interaction without cloud dependencies

**Native Multimodality:**
- Eliminated preprocessing pipelines for different modalities
- Achieved consistent quality across all input types
- Reduced system complexity and maintenance overhead

## 8. Conclusion

PData demonstrates the innovative potential of unified multimodal AI models for personal knowledge management. By leveraging Gemma 3n's native multimodal capabilities, we created a system that:

1. **Simplifies Architecture**: Single model handles all input modalities
2. **Improves Performance**: memory usage and processing speed
3. **Enhances Privacy**: Complete on-device processing
4. **Increases Accuracy**: Better cross-modal understanding through unified training

The system's success validates the effectiveness of unified multimodal models for real-world applications and demonstrates how innovative use of cutting-edge AI capabilities can create practical, privacy-preserving personal analytics solutions.

### 8.1 Future Work

1. **Mobile Deployment**: Optimize for mobile devices with model quantization
2. **Advanced Analytics**: Implement machine learning insights and pattern recognition
3. **Collaborative Features**: Enable sharing and collaboration while maintaining privacy
4. **Integration APIs**: Connect with external services for enhanced functionality

PData represents a significant step forward in personal knowledge management, showcasing how unified multimodal AI can transform the way we understand and analyze our daily lives.

---

## Appendix A: System Requirements

### Hardware Requirements
- **RAM**: Minimum 16GB, Recommended 32GB+
- **GPU**: Optional, CUDA-compatible for acceleration
- **CPU**: Multi-core processor recommended

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **FastAPI**: 0.100+
- **Uvicorn**: 0.20+

## Appendix B: Installation and Setup

```bash
# Clone repository
git clone https://github.com/Neal-wu/PData.git
cd PData

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "HF_TOKEN=your_huggingface_token" > .env

# Run the application
python main.py
```

## Appendix C: API Endpoints

### Core Endpoints
- `POST /process`: Process multimodal inputs and extract triples
- `GET /analytics`: Access personal analytics dashboard
- `GET /analytics/{category}`: Get category-specific reports
- `GET /analytics/stats`: Get overall statistics

### Response Formats
All endpoints return structured JSON responses with consistent error handling and status codes.

---

*This technical report documents the PData system as of version 1.0. For the latest updates and source code, visit the project repository.* 