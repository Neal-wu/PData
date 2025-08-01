# This project is developed specifically for kaggle competition https://www.kaggle.com/competitions/google-gemma-3n-hackathon/overview

# PData: Personalized Graph Database

PData is a comprehensive personal knowledge graph system that extracts, categorizes, and analyzes your daily activities across multiple life areas. Built with advanced LLM model Gemma 3n, it transforms your text, images, audio, and video inputs into meaningful insights and visual knowledge graphs.

## 🌟 Key Features

### 📊 **Personal Analytics Dashboard**
- **Life Category Analysis**: Automatically categorizes activities into 5 life areas:
  - 📚💼 **Education & Work**: Learning, studying, work projects, meetings
  - 🏥 **Healthcare**: Exercise, medication, sleep, diet, wellness
  - 🎮 **Entertainment**: Media consumption, hobbies, social activities
  - 💰 **Finance**: Spending, investments, banking, budgeting
  - 📋 **Other Activities**: Miscellaneous life activities

### 🔍 **Weekly Reports & Insights**
- **Smart Categorization**: AI-powered activity classification
- **Temporal Analysis**: Date-based tracking and trend analysis
- **Personalized Recommendations**: AI-generated suggestions for life balance
- **Activity Statistics**: Detailed metrics and patterns
- **Export Capabilities**: JSON export for external analysis

### 🎯 **Unified Multimodal Processing**
- **Text Input**: Using Gemma for diary entries, notes, descriptions
- **Image Analysis**: Direct visual content extraction using Gemma 3n
- **Audio Processing**: Direct audio-to-text conversion using Gemma 3n
- **Video Analysis**: Direct video content analysis using Gemma 3n

### 📈 **Interactive Knowledge Graphs**
- **Dynamic Visualization**: Interactive network graphs using Pyvis
- **Relationship Mapping**: Automatic triple extraction (Subject-Predicate-Object)
- **Real-time Updates**: Live graph generation from inputs
- **Export Options**: HTML visualization export

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 16GB+ RAM (for Gemma 3n loading)
- Hugging Face account with access to Gemma models

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PData
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the application:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`

## 📖 Usage Guide

### Main Interface
1. **Input Your Activities**: Enter text describing your daily activities
2. **Upload Media**: Optionally upload images, audio, or video files
3. **Extract & Visualize**: Click to generate your personal knowledge graph
4. **View Analytics**: Access detailed category-based reports

### Personal Analytics Dashboard
- **Overall Statistics**: View total activities, categories tracked, and most active areas
- **Category Reports**: Generate detailed weekly reports for each life area
- **Search & Filter**: Find specific activities within categories
- **Trend Analysis**: Track patterns over time

### Example Inputs
```
Text: "I went for a 30-minute walk, had a productive team meeting, 
studied Python programming for 2 hours, and paid my electricity bill."

Image: Photo of your workout session
Audio: Voice memo about your day
Video: Recording of a presentation
```

## 🏗️ Architecture

### Core Components

#### **AI Processing Pipeline**
- **Model**: Google Gemma 3n-E2B-it (unified multimodal)
- **Processing**: Single model handles text, image, audio, video natively
- **Output**: Structured knowledge triples from any input type

#### **Analytics Engine**
- **Categorization**: Keyword-based activity classification
- **Temporal Tracking**: Date-based activity recording
- **Insight Generation**: Pattern recognition and recommendations
- **Report Generation**: Automated weekly summaries

#### **Web Interface**
- **FastAPI Backend**: RESTful API with real-time processing
- **Interactive Frontend**: Modern, responsive web interface
- **Graph Visualization**: Dynamic network graphs
- **Analytics Dashboard**: Comprehensive reporting interface

### File Structure
```
PData/
├── main.py                          # FastAPI application entry point & Gemma 3n processing
├── personal_analytics.py            # Core analytics engine
├── personal_analytics_integration.py # Web interface integration
├── templates/
│   └── index.html                   # Main web interface
├── graphs/                          # Generated knowledge graph visualizations
├── frames/                          # Temporary video frame storage
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Technical Details

### Dependencies
- **FastAPI**: Modern web framework for APIs
- **Transformers**: Hugging Face AI model library
- **NetworkX**: Graph data structures and algorithms
- **Pyvis**: Interactive network visualizations
- **Gemma 3n**: Unified multimodal AI model (handles text, image, audio, video)

### AI Model Specifications
- **Model**: `google/gemma-3n-E2B-it`
- **Type**: Unified multimodal (text, image, audio, video)
- **Size**: ~2B parameters
- **Capabilities**: Native multimodal knowledge extraction, no separate preprocessing needed

## 🔒 Privacy & Security

- **Local Processing**: All AI processing happens on your device
- **No Cloud Storage**: Data never leaves your machine
- **Secure Storage**: Local file system with optional encryption
- **Model Privacy**: Downloaded models run entirely locally

## 🎯 Use Cases

### Personal Productivity
- Track daily activities and habits
- Analyze time allocation across life areas
- Identify patterns and optimize routines
- Set and monitor personal goals

### Health & Wellness
- Monitor exercise and diet patterns
- Track medication and health metrics
- Analyze sleep and wellness trends
- Generate health insights and recommendations

### Learning & Development
- Document learning activities and progress
- Track skill development over time
- Analyze study patterns and effectiveness
- Connect learning to real-world applications

### Financial Management
- Track spending patterns and habits
- Monitor investment activities
- Analyze budget compliance
- Generate financial insights

## 🚀 Future Enhancements

### Planned Features
- **Mobile App**: Native iOS/Android applications
- **Advanced Analytics**: Machine learning insights
- **Integration APIs**: Connect with external services
- **Collaborative Features**: Share insights with family/team
- **Advanced Visualization**: 3D graphs and timeline views

### Performance Optimizations
- **Model Quantization**: Further reduce memory usage
- **Caching**: Optimize repeated operations
- **Batch Processing**: Handle multiple inputs efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
- Create an issue in the repository
- Check the documentation
- Review the code comments for implementation details

---

**PData**: Transform your daily activities into meaningful insights with AI-powered personal analytics. 

