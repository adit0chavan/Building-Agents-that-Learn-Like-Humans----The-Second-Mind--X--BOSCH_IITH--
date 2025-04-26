import streamlit as st
import time
from stools import SupervisorAgent
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page config for Streamlit application
st.set_page_config(
    page_title="The Second Mind",
    page_icon="🧠",
    layout="wide"  # Use wide layout for better visualization of research data
)

# Initialize session state variables for persistent state across reruns
if 'assistant' not in st.session_state:
    # Create the supervisor agent instance (core backend)
    st.session_state.assistant = SupervisorAgent()
    
    # Make sure current_session_results is initialized
    if hasattr(st.session_state.assistant, 'current_session_results'):
        st.session_state.assistant.current_session_results = []
    else:
        setattr(st.session_state.assistant, 'current_session_results', [])
        
if 'results' not in st.session_state:
    # Store research results between UI refreshes
    st.session_state.results = None
if 'processing' not in st.session_state:
    # Track whether research is currently in progress
    st.session_state.processing = False
if 'current_step' not in st.session_state:
    # Track the current processing step for progress updates
    st.session_state.current_step = ""

# Custom CSS for styled UI components
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
    .history-card {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #ffffff;
    }
    .score-breakdown {
        display: flex;
        gap: 10px;
    }
    .score-component {
        padding: 5px;
        border-radius: 5px;
        font-size: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)

# Application header
st.title("🧠 The Second Mind")
st.markdown("""
    An AI-powered research assistant that learns and improves with each interaction.
    Enter your research query below to begin.
""")

# Display History in Sidebar - always visible by default
st.sidebar.header("Research History")

# Check if history exists and display it
if hasattr(st.session_state.assistant, 'memory_store') and st.session_state.assistant.memory_store:
    # Show most recent history items first (limited to 5)
    for i, item in enumerate(reversed(st.session_state.assistant.memory_store)):
        if i >= 5:  # Show only the 5 most recent history items
            break
            
        # Get the query text (handling both dictionary and object formats)
        if isinstance(item, dict):
            query_text = item.get('query', 'Unknown')
        else:
            query_text = getattr(item, 'query', 'Unknown')
            
        # Create expandable section for each history item
        with st.sidebar.expander(f"Query: {query_text}", expanded=(i == 0 and len(st.session_state.assistant.memory_store) == 1)):
            # Format the timestamp for readability
            try:
                if isinstance(item, dict):
                    timestamp = item.get('timestamp', '')
                else:
                    timestamp = str(getattr(item, 'timestamp', ''))
                    
                formatted_time = timestamp.split('T')[0] if 'T' in timestamp else timestamp
            except:
                formatted_time = "Unknown"
                
            st.markdown(f"**Researched on:** {formatted_time}")
            
            # Display key concepts with code-style formatting
            key_concepts = []
            if isinstance(item, dict) and 'key_concepts' in item:
                key_concepts = item['key_concepts']
            elif hasattr(item, 'key_concepts'):
                key_concepts = item.key_concepts
            
            if key_concepts:
                st.markdown("**Key Concepts:**")
                concepts_str = ", ".join([f"`{c}`" for c in key_concepts[:6]])
                st.markdown(concepts_str)
            
            # Display top sources with link
            top_sources = []
            if isinstance(item, dict) and 'top_sources' in item:
                top_sources = item['top_sources']
            elif hasattr(item, 'top_sources'):
                top_sources = item.top_sources
            
            if top_sources:
                st.markdown("**Top Sources:**")
                # Display only the first source with link
                if top_sources and len(top_sources) > 0:
                    top = top_sources[0]
                    if isinstance(top, dict):
                        st.markdown(f"[{top.get('title', 'Untitled')}]({top.get('url', '#')})")
                        st.markdown(f"Score: {top.get('score', 0):.2f}")
                    else:
                        st.markdown(f"{getattr(top, 'title', 'Untitled')}")
                
                # Instead of second source, display key concepts
                if key_concepts:
                    st.markdown("**Key Concepts:**")
                    # Display up to 5 key concepts as a comma-separated list
                    concepts_list = key_concepts[:5]
                    st.markdown(", ".join(concepts_list))
            
            # Show summary if available
            if isinstance(item, dict) and 'summary' in item and item['summary']:
                st.markdown("**Summary:**")
                st.markdown(item['summary'][:200] + "..." if len(item['summary']) > 200 else item['summary'])
            elif hasattr(item, 'summary') and item.summary:
                st.markdown("**Summary:**")
                st.markdown(item.summary[:200] + "..." if len(item.summary) > 200 else item.summary)
            
            # Show feedback if available
            if isinstance(item, dict) and 'feedback' in item and item['feedback']:
                st.markdown("**Feedback:**")
                st.markdown(f"_{item['feedback']}_")
            elif hasattr(item, 'feedback') and item.feedback:
                st.markdown("**Feedback:**")
                st.markdown(f"_{item.feedback}_")
else:
    st.sidebar.info("No research history available yet.")

# Custom process_query function to update progress
async def process_with_progress(query, progress_bar, status_text):
    """
    Process a research query with visual progress feedback.
    
    This function handles the research process while updating the UI to show progress.
    It divides the research into 5 steps and updates the progress bar and status text
    as each step completes.
    
    Parameters:
    - query: The research query input by the user
    - progress_bar: Streamlit progress bar component
    - status_text: Streamlit text component for status updates
    
    Returns:
    - The complete research results from the supervisor agent
    """
    # Calculate progress increments
    # 20% for initial query generation, 20% per iteration (3 iterations), 20% for final report
    TOTAL_STEPS = 5
    progress_increment = 1.0 / TOTAL_STEPS
    
    # Step 1: Initial query processing and preparation
    progress_bar.progress(progress_increment)
    status_text.markdown("**Step 1/5:** Generating search queries... 🔍")
    time.sleep(1)  # Small delay to allow UI update
    
    # Process iterations
    for i in range(3):  # 3 iterations
        # Step 2-4: Process each iteration
        current_progress = progress_increment * (i + 2)
        progress_bar.progress(current_progress)
        status_text.markdown(f"**Step {i+2}/5:** Processing research iteration {i+1}... 📚")
        
        # For iteration 1, generate search queries
        if i == 0:
            st.session_state.current_step = "Generating search queries"
            time.sleep(1)
        
        # For each iteration, simulate research steps with more detailed updates
        research_steps = [
            f"Searching for information on '{query}'",
            f"Analyzing academic and technical sources",
            f"Evaluating source credibility and relevance",
            f"Compiling findings for iteration {i+1}"
        ]
        
        sub_container = st.empty()
        for step in research_steps:
            st.session_state.current_step = step
            sub_container.info(f"⏳ {step}")
            time.sleep(0.5)  # Small delay to show progress
        
    # Actually process the query
    result = await st.session_state.assistant.process_query(query)
    
    # Final step: Generating reflection
    progress_bar.progress(0.95)
    status_text.markdown("**Step 5/5:** Generating final research reflection... 🧠")
    time.sleep(1)
    
    # Complete
    progress_bar.progress(1.0)
    status_text.markdown("**✅ Research completed successfully!**")
    
    return result

# Helper function to run async functions in Streamlit
def run_async(coroutine):
    """
    Utility function to run asynchronous code in Streamlit.
    
    Parameters:
    - coroutine: The asynchronous function to run
    
    Returns:
    - The result of the coroutine
    """
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result

# Query Input with suggestions from history
query = st.text_input("Enter your research query:", placeholder="e.g., Renewable energy for urban areas")

# Show related history suggestions if available
if query and hasattr(st.session_state.assistant, 'memory_store') and st.session_state.assistant.memory_store:
    # Check for similar queries in history
    related_history = []
    for item in st.session_state.assistant.memory_store:
        # Get query and key concepts regardless of format
        if isinstance(item, dict):
            item_query = item.get('query', '').lower()
            item_concepts = item.get('key_concepts', [])
        else:
            item_query = getattr(item, 'query', '').lower()
            item_concepts = getattr(item, 'key_concepts', [])
            
        # Check if current query is related to this item
        if (query.lower() in item_query or 
            any(query.lower() in concept.lower() for concept in item_concepts if isinstance(concept, str))):
            related_history.append(item)
    
    if related_history:
        st.markdown("**Related to your previous research:**")
        cols = st.columns(min(3, len(related_history)))
        for i, item in enumerate(related_history):
            if i < len(cols):
                with cols[i]:
                    # Get item details regardless of format
                    if isinstance(item, dict):
                        item_query = item.get('query', '')
                        top_sources = item.get('top_sources', [])
                    else:
                        item_query = getattr(item, 'query', '')
                        top_sources = getattr(item, 'top_sources', [])
                    
                    # Get the top source if available
                    top_title = ""
                    if top_sources:
                        top_source = top_sources[0]
                        if isinstance(top_source, dict):
                            top_title = top_source.get('title', '')
                        else:
                            top_title = getattr(top_source, 'title', '')
                    
                    st.info(f"**{item_query}**\n{top_title}")

# Process Button
if st.button("Start Research", type="primary") and query:
    st.session_state.processing = True
    st.session_state.results = None
    
    # Create progress bar and status containers
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_detail = st.empty()
    
    try:
        # Use our custom processing function with progress updates
        result = run_async(process_with_progress(query, progress_bar, status_text))
        st.session_state.results = result
        st.session_state.processing = False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.processing = False
        progress_bar.progress(0)
        status_text.error("Research failed")

# Display Results
if st.session_state.results:
    # Check if there was an error during processing
    if "error" in st.session_state.results:
        st.error(f"An error occurred during research: {st.session_state.results['error']}")
    else:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Final Results", "Iteration Details", "Sources", "Raw Data"])
        
        with tab1:
            st.header("Final Research Report")
            if "meta_review" in st.session_state.results:
                meta_review = st.session_state.results["meta_review"]
                
                # Display evolution summary
                st.subheader("Research Evolution")
                st.markdown(meta_review.get("evolution_summary", "No evolution summary available."))
                
                # Display consistent findings
                st.subheader("Key Findings")
                for finding in meta_review.get("consistent_findings", []):
                    st.markdown(f"- {finding}")
                
                # Display best solutions
                st.subheader("Best Solutions")
                for solution in meta_review.get("best_solutions", []):
                    st.markdown(f"- {solution}")
                
                # Display next steps
                st.subheader("Recommended Next Steps")
                for step in meta_review.get("next_steps", []):
                    st.markdown(f"- {step}")
            else:
                st.warning("Meta-review is not available.")
        
        with tab2:
            st.header("Research Iterations")
            if "cycles" in st.session_state.results:
                for idx, cycle in enumerate(st.session_state.results["cycles"]):
                    with st.expander(f"Iteration {idx + 1}", expanded=(idx == 0)):
                        # Display hypothesis
                        st.markdown(f"**Hypothesis:** {cycle.get('hypothesis', 'N/A')}")
                        
                        # Display reflection/analysis
                        st.markdown("**Analysis:**")
                        if "reflection" in cycle and "analysis" in cycle["reflection"]:
                            st.markdown(cycle["reflection"]["analysis"])
                        else:
                            st.warning("Analysis information is not available.")
                        
                        # Display key concepts
                        st.markdown("**Key Concepts:**")
                        if "reflection" in cycle and "key_concepts" in cycle["reflection"]:
                            for concept in cycle["reflection"]["key_concepts"]:
                                st.markdown(f"- {concept}")
                        else:
                            st.warning("Key concepts are not available.")
                        
                        # Display trends if available
                        if "reflection" in cycle and "trends" in cycle["reflection"] and cycle["reflection"]["trends"]:
                            st.markdown("**Emerging Trends:**")
                            for trend in cycle["reflection"]["trends"]:
                                st.markdown(f"- {trend}")
                        
                        # Display gaps if available
                        if "reflection" in cycle and "gaps" in cycle["reflection"] and cycle["reflection"]["gaps"]:
                            st.markdown("**Research Gaps:**")
                            for gap in cycle["reflection"]["gaps"]:
                                st.markdown(f"- {gap}")
            else:
                st.warning("Iteration information is not available.")
        
        with tab3:
            st.header("Sources")
            if "cycles" in st.session_state.results:
                for idx, cycle in enumerate(st.session_state.results["cycles"]):
                    st.subheader(f"Iteration {idx + 1} Sources")
                    
                    if "ranking" in cycle and "ranked_sources" in cycle["ranking"]:
                        for i, source in enumerate(cycle["ranking"]["ranked_sources"][:5]):
                            with st.expander(f"{i+1}. {source.get('title', 'Untitled')} (Score: {source.get('overall_score', 0):.2f})"):
                                st.markdown(f"**URL:** [{source.get('url', 'N/A')}]({source.get('url', '#')})")
                                st.markdown(f"**Type:** {source.get('source_type', 'Unknown').capitalize()}")
                                st.markdown(f"**Publication Date:** {source.get('pub_date', 'Unknown')}")
                                st.markdown(f"**Citation Count:** {source.get('citation_count', 0)}")
                                
                                # Display score breakdown if available
                                if "score_breakdown" in source:
                                    st.markdown("**Score Breakdown:**")
                                    breakdown = source["score_breakdown"]
                                    
                                    # Create visual breakdown using columns
                                    cols = st.columns(len(breakdown))
                                    for j, (factor, score) in enumerate(breakdown.items()):
                                        with cols[j]:
                                            st.markdown(f"**{factor.replace('_', ' ').title()}**")
                                            st.progress(float(score))
                                            st.markdown(f"{float(score):.2f}")
                                
                                # Display content snippet
                                if "content" in source and source["content"]:
                                    st.markdown("**Content Snippet:**")
                                    st.markdown(f"```\n{source['content'][:300]}...\n```")
                                
                                # Display key insights
                                if "key_insights" in source and source["key_insights"]:
                                    st.markdown("**Key Insights:**")
                                    for insight in source["key_insights"]:
                                        st.markdown(f"- {insight}")
                    else:
                        st.warning("No ranked sources available for this iteration.")
            else:
                st.warning("Iteration information is not available.")
                
        with tab4:
            st.header("Raw Data")
            st.json(st.session_state.results)

# Refinement options if results are available
if st.session_state.results and "error" not in st.session_state.results:
    st.markdown("---")
    st.subheader("Refine Your Research")
    
    refine_cols = st.columns(3)
    with refine_cols[0]:
        # Option for providing feedback
        feedback = st.text_area("Provide feedback to improve the research", placeholder="Enter your feedback here...")
        if st.button("Submit Feedback", type="secondary") and feedback:
            st.session_state.processing = True
            
            # Create progress bar and status containers
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Show initial progress
                progress_bar.progress(0.2)
                status_text.markdown("**Processing feedback...**")
                
                try:
                    # Call the process_query with feedback
                    progress_bar.progress(0.4)
                    status_text.markdown("**Incorporating feedback into research...**")
                    updated_results = run_async(st.session_state.assistant.process_query(query, feedback=feedback))
                    
                    progress_bar.progress(0.8)
                    status_text.markdown("**Finalizing updated research...**")
                    
                    st.session_state.results = updated_results
                    st.session_state.processing = False
                    
                    progress_bar.progress(1.0)
                    status_text.markdown("**✅ Research updated successfully!**")
                    time.sleep(1)  # Show success message briefly
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing feedback: {str(e)}")
                    progress_bar.progress(0)
                    status_text.error("Failed to process feedback")
                    st.session_state.processing = False
    
    with refine_cols[1]:
        # Option for deep research
        deep_research_topics = st.text_input("Enter topics for deep research (comma-separated)", placeholder="e.g., solar panels, urban planning, energy storage")
        if st.button("Deep Research", type="secondary") and deep_research_topics:
            topics = [topic.strip() for topic in deep_research_topics.split(",") if topic.strip()]
            if topics:
                st.session_state.processing = True
                
                # Create progress bar and status containers
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show initial progress
                    progress_bar.progress(0.2)
                    status_text.markdown(f"**Starting deep research on: {', '.join(topics)}**")
                    
                    try:
                        # Call the process_query with deep_research topics
                        for i, topic in enumerate(topics):
                            # Update progress for each topic
                            progress = 0.2 + (0.6 * (i / len(topics)))
                            progress_bar.progress(progress)
                            status_text.markdown(f"**Researching topic {i+1}/{len(topics)}: {topic}**")
                            time.sleep(0.5)  # Small visual delay
                        
                        deep_results = run_async(st.session_state.assistant.process_query(query, deep_research=topics))
                        
                        progress_bar.progress(0.9)
                        status_text.markdown("**Synthesizing deep research findings...**")
                        
                        st.session_state.results = deep_results
                        st.session_state.processing = False
                        
                        progress_bar.progress(1.0)
                        status_text.markdown("**✅ Deep research completed successfully!**")
                        time.sleep(1)  # Show success message briefly
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in deep research: {str(e)}")
                        progress_bar.progress(0)
                        status_text.error("Deep research failed")
                        st.session_state.processing = False
            else:
                st.warning("Please enter valid topics for deep research.")
    
    with refine_cols[2]:
        if st.button("New Research", type="secondary"):
            # Reset the research agent completely
            st.session_state.assistant.reset()
            
            # Get history file path from environment variables
            history_file = os.getenv("HISTORY_FILE", "research_history.json")
            feedback_file = os.getenv("FEEDBACK_FILE", "feedback_history.json")
            
            # Additional step to ensure history JSON file is cleared
            try:
                if os.path.exists(history_file):
                    with open(history_file, "w") as f:
                        f.write("[]")  # Write empty array to JSON file
                if os.path.exists(feedback_file):
                    with open(feedback_file, "w") as f:
                        f.write("[]")  # Clear feedback history too
            except Exception as e:
                st.error(f"Error clearing history files: {str(e)}")
            
            # Reinitialize the assistant for a completely fresh start
            st.session_state.assistant = SupervisorAgent()
            
            # Make sure current_session_results is reset
            if hasattr(st.session_state.assistant, 'current_session_results'):
                st.session_state.assistant.current_session_results = []
            
            # Clear all session state variables
            st.session_state.results = None
            st.session_state.processing = False
            st.session_state.current_step = ""
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and The Second Mind") 