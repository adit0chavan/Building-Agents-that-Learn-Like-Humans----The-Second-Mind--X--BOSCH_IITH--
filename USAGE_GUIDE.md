# 🧠 The Second Mind - Detailed Usage Guide

This guide provides step-by-step instructions on how to use The Second Mind research assistant effectively, including examples and best practices.

## 🚀 Getting Started

### Initial Setup

1. Make sure you have installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your `.env` file contains valid API keys:
   ```
   GROQ_API_KEY="your_groq_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

3. Launch the application:
   ```bash
   streamlit run app.py
   ```

4. Open your web browser and navigate to http://localhost:8501

## 📝 Conducting Research

### Formulating Effective Queries

For best results, craft your research queries to be:

- **Specific**: "What are the latest advancements in quantum computing error correction?" instead of "quantum computing"
- **Focused**: Target a particular aspect of a broader topic
- **Contextual**: Include relevant domain information if applicable

Examples of effective queries:
- "What are the environmental impacts of lithium mining for EV batteries?"
- "Compare machine learning approaches for natural language processing in healthcare"
- "Current challenges in sustainable urban planning for growing cities"

### Starting the Research Process

1. Enter your research query in the main text field at the top of the page
2. Review any related previous research suggestions that may appear
3. Click the "Start Research" button to begin the process
4. Watch the progress indicators as the system works through multiple research iterations

### Understanding the Progress Indicators

During the research process, you'll see:
- A progress bar showing overall completion
- Status text describing the current stage
- Step details showing specific actions being taken

The typical research process takes 3-5 minutes to complete depending on query complexity.

## 📊 Exploring Research Results

### Final Results Tab

The primary tab shows the high-level synthesis of all research:

1. **Research Evolution**: How the research progressed across multiple iterations
2. **Key Findings**: The most important and consistent discoveries
3. **Best Solutions**: Practical applications or solutions identified
4. **Recommended Next Steps**: Suggested areas for further investigation

This tab is ideal for getting a quick overview of the research outcomes.

### Iteration Details Tab

This tab provides deeper insight into the research process:

1. Each iteration is shown as an expandable section
2. For each iteration, you can view:
   - **Hypothesis**: The specific question being investigated
   - **Analysis**: Detailed interpretation of findings
   - **Key Concepts**: Important terms and ideas identified
   - **Emerging Trends**: Patterns identified in the research
   - **Research Gaps**: Areas where information is limited or contradictory

Use this tab to understand how the research evolved and refined over time.

### Sources Tab

This tab offers comprehensive information about all sources used:

1. Sources are grouped by research iteration
2. Each source is ranked by credibility and relevance
3. For each source, you can view:
   - **Title and URL**: Direct link to the source
   - **Source Type**: Academic or general
   - **Publication Date**: When the source was published
   - **Citation Count**: Number of citations (if available)
   - **Score Breakdown**: Visual representation of quality metrics
   - **Content Snippet**: Brief excerpt from the source
   - **Key Insights**: Important information extracted from the source

This tab is valuable for evaluating the quality of the research and accessing primary sources.

### Raw Data Tab

This tab displays the complete JSON structure of all research results for technical users or those who want to see every detail of the research process.

## 🔄 Refining Your Research

### Providing Feedback

To improve results or focus on specific aspects:

1. Scroll to the "Refine Your Research" section
2. Enter your feedback in the text area
3. Click "Submit Feedback"
4. The system will incorporate your feedback and regenerate results

Effective feedback examples:
- "Please focus more on practical applications rather than theoretical concepts"
- "The sources seem outdated. Can you find more recent research?"
- "I need more technical depth in the analysis of machine learning algorithms"

### Conducting Deep Research

To explore specific subtopics in greater detail:

1. Identify key subtopics from your initial research results
2. Enter these topics (comma-separated) in the "Deep Research" input field
3. Click "Deep Research"
4. The system will perform specialized research on each subtopic
5. Results will include deeper analysis and connections between subtopics

Example deep research topics for a query about renewable energy:
- "solar panel efficiency, energy storage solutions, grid integration challenges"

### Starting Fresh Research

When you want to begin a completely new research topic:

1. Click "New Research" to reset the system
2. This clears all current research data and history
3. Enter your new query and begin the process again

## 📚 Learning From History

The system maintains a history of your research sessions:

1. Previous research topics appear in the sidebar
2. Click on any previous research to expand details
3. Use related history suggestions when they appear above the query field
4. The system learns from past research to provide better results over time

## 💡 Best Practices

- **Be patient** during the research process - quality analysis takes time
- **Start broad** then use deep research for specific aspects
- **Provide constructive feedback** to help the system learn your preferences
- **Compare sources** in the Sources tab to evaluate information quality
- **Use multiple research sessions** on related topics to build comprehensive knowledge

## 🔍 Example Research Workflow

Here's a complete example workflow:

1. Enter query: "What are the most effective sustainable agriculture practices for small farms?"
2. Review initial results focusing on the Final Results tab
3. Identify key subtopics: "water conservation techniques, organic pest management, soil health optimization"
4. Conduct deep research on these subtopics
5. Provide feedback: "Please focus more on low-cost solutions applicable to developing countries"
6. Review refined results
7. Export or save key findings for implementation

## 🛠️ Troubleshooting

- If research seems stuck, refresh the page and try again
- If results lack depth, try reformulating your query to be more specific
- For technical errors, check your API keys and internet connection
- For missing or incomplete sources, try deep research on specific subtopics 