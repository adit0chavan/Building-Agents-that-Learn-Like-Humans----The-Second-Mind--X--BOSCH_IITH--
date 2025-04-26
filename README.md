# 🧠 The Second Mind

A sophisticated AI research assistant system that mimics human learning by retaining preferences, connecting ideas across topics, and continuously improving with each interaction.

> **🏆 Recognition**: This project was developed for the Bosch x IITH competition and was selected as one of the top 20 teams out of more than 300 participating teams.

## 👥 Team Members

- **Aditya Chavan** (Team Leader) - [LinkedIn](https://www.linkedin.com/in/aditya-chavan-5117aa268/)
- **Saumya Shah** - [LinkedIn](https://www.linkedin.com/in/saumya-shah-9b2579273/)
- **Bhavin Baldota** - [LinkedIn](https://www.linkedin.com/in/bhavin-baldota-103553234/)

## 📋 Overview

The Second Mind is a coalition of specialized AI agents that work together to perform comprehensive research on any topic. The system leverages Large Language Models (LLMs) and real-time web data extraction to create a research experience that becomes increasingly personalized and intelligent over time.

Unlike traditional search engines or AI assistants, The Second Mind implements a multi-agent, iterative research process that refines outputs through multiple cycles. It analyzes source credibility, extracts key concepts, identifies trends and gaps, and synthesizes findings into coherent, insightful research reports.

## 🎯 Problem Statement

The challenge was to develop a system of AI agents that mimics human learning—retaining preferences, connecting ideas, and improving with each interaction. The goal was to implement a coalition of specialized agents that iteratively refine outputs toward a research goal, incorporating real-time web data extraction for enhanced research capabilities.

## 🌟 Key Features

### Core Capabilities

- **Iterative Research Process**: Conducts multiple research cycles to progressively refine findings
- **Source Credibility Analysis**: Evaluates sources based on relevance, recency, citation count, and content quality
- **Memory Retention**: Stores previous research sessions and leverages them for future queries
- **Feedback Integration**: Learns from user feedback to improve future research quality
- **Deep Research**: Performs specialized in-depth analysis on specific subtopics
- **Cross-Topic Synthesis**: Connects insights across different research domains

### Technical Components

- **Multi-Agent Architecture**: Coalition of specialized agents coordinated by a supervisor
- **Real-Time Web Data Extraction**: Integration with Tavily API for academic and general source retrieval
- **Dynamic Source Ranking**: Complex algorithm to score and rank sources using multiple weighted factors
- **Persistent Memory**: JSON-based storage of research history with intelligent retrieval
- **Asynchronous Processing**: Non-blocking research workflow using Python's asyncio
- **Interactive Progress Tracking**: Real-time feedback on research progress

### User Interface

- **Modern Streamlit Interface**: Clean, responsive web application with intuitive layout
- **Research History Sidebar**: Quick access to previous research sessions
- **Multi-Tab Results View**: Organized presentation of findings, iterations, sources, and raw data
- **Visual Source Scoring**: Graphical representation of source credibility factors
- **Research Refinement Tools**: Options for feedback submission and deep research requests

## 🛠 Technical Architecture

### Agent System

- **SupervisorAgent**: Central controller that orchestrates the entire research process
- **Web Search Tool**: Retrieves and categorizes sources from the internet using Tavily API
- **Analysis Tool**: Processes findings to extract key concepts, trends, and research gaps
- **Ranking Tool**: Evaluates source credibility and relevance using a multi-factor algorithm
- **Hypothesis Evolution Tool**: Refines research approach based on intermediate findings
- **Feedback Integration Tool**: Incorporates user feedback to improve research quality
- **Deep Research Tool**: Conducts specialized research on specific subtopics

### Core Technologies

- **LangChain Framework**: Used for agent creation, tool integration, and LLM interaction
- **Groq LLM Integration**: Powers the intelligent reasoning and natural language understanding
- **Tavily Search API**: Enables real-time web data extraction with academic focus
- **Streamlit**: Provides the interactive web application interface
- **Python Asyncio**: Enables non-blocking concurrent research operations
- **JSON-based Persistence**: Maintains research history and user feedback between sessions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Groq (LLM access)
  - Tavily (web search capabilities)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/second-mind.git
   cd second-mind
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY="your_groq_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   MODEL_NAME="llama-3.3-70b-versatile"
   TEMPERATURE=0.7
   MAX_HISTORY=10
   DEFAULT_ITERATIONS=3
   HISTORY_FILE="research_history.json"
   FEEDBACK_FILE="feedback_history.json"
   ```

### Running the Application

Launch the application with:
```bash
streamlit run app.py
```

The system will be available at http://localhost:8501 in your web browser.

## 💡 Usage Guide

### Starting Research

1. Enter your research query in the main input field
2. Click "Start Research" to begin the multi-iteration research process
3. Track progress through the visual progress bar and status updates

### Exploring Results

Results are organized into four tabs:

1. **Final Results**: Overall synthesis with research evolution, key findings, best solutions, and recommended next steps
2. **Iteration Details**: Breakdown of each research cycle with hypothesis, analysis, key concepts, trends, and research gaps
3. **Sources**: Ranked list of sources with detailed credibility metrics, content snippets, and key insights
4. **Raw Data**: Complete JSON data for technical inspection

### Refining Research

After reviewing results, you can:

- **Provide Feedback**: Submit comments to improve the quality of findings
- **Request Deep Research**: Specify subtopics for more focused, in-depth analysis
- **Start New Research**: Clear all current findings to begin a new topic

## 🔧 Implementation Details

### Research Process Flow

1. **Query Processing**: Initial analysis of the research question
2. **Web Search**: Multiple search queries with academic focus
3. **Source Analysis**: Categorization and credibility evaluation
4. **Content Extraction**: Identification of key concepts and insights
5. **Multi-Iteration Refinement**: Three research cycles by default
6. **Meta-Review Generation**: Cross-iteration synthesis and recommendation
7. **Result Presentation**: Organized display with different detail levels

### Memory and Learning System

- **Research History**: Persistent storage of previous sessions
- **Cross-Query Learning**: Ability to connect related research topics
- **Feedback Integration**: Continuous improvement from user comments
- **Query Refinement**: Automatic query evolution based on intermediate findings

## 🧪 Project Structure

- `app.py`: Streamlit web application and user interface
- `stools.py`: Core research engine with SupervisorAgent implementation
- `requirements.txt`: Required Python packages
- `research_history.json`: Persistent storage of research history
- `feedback_history.json`: Storage for user feedback

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit, LangChain, and Groq LLM
- Uses Tavily for intelligent web search capabilities

## 📸 Screenshots & Results

Here are some screenshots demonstrating the system's functionality and outputs:

### Final Research Results

The comprehensive synthesis of all research findings, including key insights, recommended solutions, and next steps:

![Final Research Results](Output_Images/Final%20Result%20Screenshot.png)

### Iteration Insights

Detailed breakdown of the research iterations, showing how the system refines its understanding:

![Iteration Insights](Output_Images/Iteration%20insights.png)
![Iteration Insights 2](Output_Images/iteration%20insights%202.png)

### Related Topics & Suggestions

The system identifies related research topics from your history:

![Related Topics](Output_Images/Related%20Topics%20GUI.png)

### User Feedback Integration

Results after incorporating user feedback, showing how the system learns:

![User Feedback Results](Output_Images/User%20feedback%20result.png)

### Research History & Feedback Interface

The interface for accessing research history and providing feedback:

![Research History & Feedback](Output_Images/User%20Feedback%20and%20history%20GUI.png)

### Raw Data Analysis

Behind-the-scenes raw data processing that powers the research:

![Raw Data Analysis](Output_Images/Raw%20Data%20\(Backend%20Query\).png) 