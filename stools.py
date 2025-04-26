import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from tavily import TavilyClient
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants loaded from environment variables
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
DEFAULT_ITERATIONS = int(os.getenv("DEFAULT_ITERATIONS", "3"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "research_history.json")
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback_history.json")

# Domain categorization for source classification
# Used to categorize search results into academic or general sources
SOURCES = {
    "academic": [
        "arxiv.org", "link.springer.com", "ieeexplore.ieee.org",
        "aclanthology.org", "dl.acm.org", "researchgate.net",
        "semanticscholar.org", "sciencedirect.com", "academic.oup.com"
    ],
    "general": [
        "towardsdatascience.com", "techcrunch.com", "wired.com",
        "mit.edu", "analyticsindiamag.com", "ai.googleblog.com"
    ]
}

@dataclass
class ResearchMemory:
    """
    Data class to store research session information.
    
    This class represents a single research session and stores various elements
    such as the original query, hypothesis, score, and research findings.
    
    Attributes:
        query: The original research query
        hypothesis: The current hypothesis or research approach
        score: A numeric score for the research quality (0-1)
        timestamp: When the research was conducted
        web_data: Dictionary containing web search results
        key_concepts: List of key concepts identified in the research
        related_queries: List of related queries that could be explored
        feedback: Optional user feedback on the research
        deep_research_topics: Optional list of topics for deeper research
    """
    query: str
    hypothesis: str
    score: float
    timestamp: datetime
    web_data: Dict[str, Any]
    key_concepts: List[str]
    related_queries: List[str]
    feedback: Optional[str] = None
    deep_research_topics: Optional[List[str]] = None

class SupervisorAgent:
    """
    Main research supervisor agent that coordinates the research process.
    
    This class is responsible for managing the entire research workflow, from
    initial query processing to final report generation. It orchestrates various
    tools and components to perform multi-iteration research with source evaluation.
    """
    def __init__(self):
        """Initialize the supervisor agent with necessary components."""
        # Initialize LLM with API key from environment variables
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        )
        
        # Initialize Tavily client for web search
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Load history from file
        self.memory_store: List[ResearchMemory] = []
        self._load_history()
        
        # Initialize session-level collection for iteration results
        self.current_session_results = []

    def _initialize_tools(self) -> List[Tool]:
        """
        Initialize all available tools for the supervisor agent.
        
        This method creates a list of LangChain Tool objects that the agent can use
        to perform various research tasks such as web search, analysis, and source ranking.
        
        Returns:
            A list of Tool objects with their name, function, and description
        """
        return [
            Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for information on a topic"
            ),
            Tool(
                name="analyze_findings",
                func=self._analyze_findings,
                description="Analyze research findings and extract key insights"
            ),
            Tool(
                name="rank_sources",
                func=self._rank_sources,
                description="Rank sources based on relevance and credibility"
            ),
            Tool(
                name="evolve_hypothesis",
                func=self._evolve_hypothesis,
                description="Evolve research hypothesis based on findings"
            ),
            Tool(
                name="check_proximity",
                func=self._check_proximity,
                description="Check similarity with past research"
            ),
            Tool(
                name="deep_research",
                func=self._deep_research,
                description="Perform deep research on specific topics"
            ),
            Tool(
                name="get_feedback",
                func=self._get_feedback,
                description="Get user feedback on research findings"
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """
        Create the supervisor agent with reasoning capabilities.
        
        This method sets up a LangChain ReAct agent with the LLM and tools,
        configuring it with a prompt that guides the research process.
        
        Returns:
            An AgentExecutor that can run the agent with tools and memory
        """
        prompt = PromptTemplate.from_template("""
        You are a research supervisor agent. Your task is to help with research by using the available tools.

        Available tools:
        {tools}

        Tool names: {tool_names}

        User query: {input}
        Previous context: {chat_history}

        {agent_scratchpad}

        Think step by step and use the tools to help with the research.
        First, use the web_search tool to find information.
        Then, use analyze_findings to process the results.
        Finally, use rank_sources to organize the findings.

        Return your response in this format:
        Action: tool_name
        Action Input: {{"query": "user's query"}}
        """)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor with memory and error handling
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )

    async def process_query(self, query: str, feedback: Optional[str] = None, 
                          deep_research: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a research query with optional feedback and deep research topics.
        
        This is the main entry point for the research process. It handles the 
        entire research workflow, from initial query to final report generation.
        
        Parameters:
            query: The research query from the user
            feedback: Optional feedback to refine the research approach
            deep_research: Optional list of topics for deep research
            
        Returns:
            A dictionary containing the complete research results
        """
        try:
            print("\nStarting research process...")
            
            # Clear previous session results
            self.current_session_results = []
            
            # Initialize results
            results = []
            
            # If feedback provided, incorporate it and modify the query approach
            if feedback:
                print("Incorporating feedback...")
                self._incorporate_feedback(feedback)
                
                # Use feedback to refine research approach
                refined_approach = await self._refine_query_with_feedback(query, feedback)
                refined_query = refined_approach.get("refined_query", query)
                focus_areas = refined_approach.get("focus_areas", [])
                
                print(f"Original query: {query}")
                print(f"Refined query: {refined_query}")
                if focus_areas:
                    print(f"Focus areas based on feedback: {', '.join(focus_areas)}")
                
                # Use the refined query for research
                query = refined_query
            
            # If deep research topics provided, focus on those
            if deep_research:
                print("Performing deep research...")
                return await self._perform_deep_research(query, deep_research)
            
            # Main research loop - run multiple iterations to refine results
            for iteration in range(DEFAULT_ITERATIONS):
                print(f"\nStarting iteration {iteration + 1} of {DEFAULT_ITERATIONS}")
                
                try:
                    # First, perform web search
                    print("Performing web search...")
                    web_results = await self._web_search({"query": query})
                    print(f"Web search completed. Found {len(web_results.get('academic', [])) + len(web_results.get('general', []))} articles")
                    
                    # Then analyze findings
                    print("\nAnalyzing findings...")
                    analysis = await self._analyze_findings({"findings": web_results})
                    print("Analysis completed")
                    
                    # Finally rank sources
                    print("\nRanking sources...")
                    ranking = await self._rank_sources({
                        "topic": query,
                        "sources": web_results
                    })
                    print("Ranking completed")
                    
                    iteration_results = {
                        "cycle": iteration + 1,
                        "web_data": web_results,
                        "reflection": analysis,  # Changed from analysis to reflection
                        "ranking": ranking,
                        "hypothesis": query  # Initial hypothesis is the query
                    }
                    
                    results.append(iteration_results)
                    
                    # Store in session results but don't save to file yet
                    print("\nCollecting results for this iteration...")
                    self._store_results(query, iteration_results)
                    
                except Exception as e:
                    print(f"Error in iteration {iteration + 1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Generate final meta-review
            print("\nGenerating meta-review...")
            meta_review = await self._generate_meta_review(results)
            
            # Now save the combined results to history
            print("\nSaving combined session results to history...")
            self._save_combined_session_results(query, meta_review)
            
            print("Research process completed successfully")
            return {
                "cycles": results,
                "meta_review": meta_review,
                "options": {
                    "feedback": "Provide feedback to improve the research",
                    "deep_research": "Request deep research on specific topics",
                    "reset": "Reset the research and start a new topic"
                }
            }
            
        except Exception as e:
            print(f"Error in research process: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def _execute_plan(self, plan: Dict, query: str) -> Dict[str, Any]:
        """Execute the supervisor agent's plan."""
        try:
            print(f"Executing plan: {json.dumps(plan, indent=2)}")
            results = {}
            
            # Execute each step in the plan
            for step in plan.get("plan", []):
                tool_name = step.get("tool")
                tool_input = step.get("input")
                
                if not tool_name or not tool_input:
                    print(f"Skipping invalid step: {step}")
                    continue
                
                print(f"Executing tool: {tool_name}")
                tool_result = await self._execute_tool(tool_name, tool_input)
                results[tool_name] = tool_result
            
            # Execute next action if specified
            if "next_action" in plan:
                next_action = plan["next_action"]
                tool_name = next_action.get("tool")
                tool_input = next_action.get("input")
                
                if tool_name and tool_input:
                    print(f"Executing next action: {tool_name}")
                    tool_result = await self._execute_tool(tool_name, tool_input)
                    results[f"next_{tool_name}"] = tool_result
            
            return results
            
        except Exception as e:
            print(f"Error executing plan: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict[str, Any]:
        """Execute a specific tool based on the supervisor's decision."""
        try:
            print(f"Looking for tool: {tool_name}")
            tool = next((tool for tool in self.tools if tool.name == tool_name), None)
            if not tool:
                print(f"Tool not found: {tool_name}")
                return {"error": f"Tool {tool_name} not found"}
            
            print(f"Executing tool {tool_name} with input: {tool_input}")
            result = await tool.func(tool_input)
            print(f"Tool {tool_name} execution completed")
            return result
            
        except Exception as e:
            print(f"Error executing tool {tool_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _process_sources(self, search_response: Dict) -> Dict:
        """
        Process search results and categorize sources into academic and general.
        
        This method takes raw search results from Tavily and processes them to:
        1. Extract key information like title, URL, and content
        2. Categorize sources as academic or general based on domain
        3. Extract additional metadata like publication date and citation count
        4. Remove duplicate sources
        
        Parameters:
            search_response: Raw search results from Tavily API
            
        Returns:
            Dictionary with sources categorized into 'academic' and 'general'
        """
        try:
            if not search_response:
                return {"academic": [], "general": []}
                
            # Get the results array from the response
            results = search_response.get('results', [])
            if not results:
                return {"academic": [], "general": []}
                
            categorized = {"academic": [], "general": []}
            seen_urls = set()
            
            for source in results:
                try:
                    # Extract basic information
                    url = source.get('url', '')
                    if not url or url in seen_urls:
                        continue
                        
                    seen_urls.add(url)
                    
                    # Get title and content
                    title = source.get('title', 'Untitled')
                    if isinstance(title, str):
                        title = title.split(' - ')[0].strip()
                    
                    content = source.get('content', '')
                    if isinstance(content, str):
                        content = content[:300] + '...' if len(content) > 300 else content
                    
                    # Determine source type
                    source_type = "academic" if any(domain in url for domain in SOURCES["academic"]) else "general"
                    
                    # Create source data
                    source_data = {
                        "title": title,
                        "url": url,
                        "content": content,
                        "pub_date": self._extract_publication_date(source),
                        "citation_count": self._extract_citation_count(source),
                        "source_type": source_type,
                        "score": 0.5  # Default score
                    }
                    
                    categorized[source_type].append(source_data)
                except Exception as e:
                    print(f"Error processing individual source: {str(e)}")
                    continue
                    
            return categorized
        except Exception as e:
            print(f"Error in source processing: {str(e)}")
            return {"academic": [], "general": []}

    def _extract_publication_date(self, source: Dict) -> str:
        """Extract publication date from source content."""
        try:
            content = source.get('content', '')
            if not isinstance(content, str):
                return ""
                
            date_patterns = [
                r'published(?:\s+on)?\s+(\w+\s+\d{1,2},?\s+\d{4})',
                r'(?:date|published):\s+(\w+\s+\d{1,2},?\s+\d{4})',
                r'(\d{4}-\d{2}-\d{2})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1)
                    
            return ""
        except Exception:
            return ""

    def _extract_citation_count(self, source: Dict) -> int:
        """Extract citation count from source content."""
        try:
            content = source.get('content', '')
            if not isinstance(content, str):
                return 0
                
            citation_patterns = [
                r'cited by (\d+)',
                r'citations?:\s*(\d+)',
                r'(\d+) citations'
            ]
            
            for pattern in citation_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
            
        return 0

    async def _web_search(self, input_data: Dict) -> Dict[str, Any]:
        """
        Perform web search using Tavily with improved extraction.
        
        This method searches the web for information on the given query using 
        the Tavily search API. It performs multiple searches with different
        query variations to maximize the diversity and quality of results.
        Results are categorized into academic and general sources.
        
        Parameters:
            input_data: Dictionary containing the 'query' parameter
            
        Returns:
            Dictionary with categorized search results
        """
        try:
            query = input_data.get("query", "")
            print(f"Searching for: {query}")
            
            # Generate multiple queries based on the input query
            technical_queries = [
                f"{query} research paper",
                f"{query} latest techniques",
                f"{query} methodologies filetype:pdf",
                f"{query} academic review"
            ]
            
            all_results = {"academic": [], "general": []}
            search_count = 0
            
            # Perform academic search with primary query
            academic_results = self.tavily.search(
                query=f"{query} filetype:pdf OR research paper",
                search_depth="advanced",
                max_results=8  # Increased from 5
            )
            
            # Process academic results
            academic_processed = self._process_sources(academic_results)
            all_results["academic"].extend(academic_processed.get("academic", []))
            all_results["general"].extend(academic_processed.get("general", []))
            search_count += 1
            
            # Perform additional searches with technical queries
            for tech_query in technical_queries:
                if search_count >= 3:  # Limit to 3 searches to avoid rate limits
                    break
                    
                try:
                    technical_results = self.tavily.search(
                        query=tech_query,
                        search_depth="advanced",
                        max_results=5
                    )
                    
                    if technical_results:
                        tech_processed = self._process_sources(technical_results)
                        all_results["academic"].extend(tech_processed.get("academic", []))
                        all_results["general"].extend(tech_processed.get("general", []))
                        search_count += 1
                except Exception as e:
                    print(f"Error in additional search: {str(e)}")
                    continue
            
            # Deduplicate results based on URL
            seen_urls = set()
            deduplicated = {"academic": [], "general": []}
            
            for source_type in ["academic", "general"]:
                for source in all_results[source_type]:
                    url = source.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        deduplicated[source_type].append(source)
            
            total_results = len(deduplicated["academic"]) + len(deduplicated["general"])
            print(f"Total sources found after deduplication: {total_results}")
            
            return deduplicated
            
        except Exception as e:
            print(f"Error in web search: {str(e)}")
            return {"academic": [], "general": []}

    async def _analyze_findings(self, input_data: Dict) -> Dict[str, Any]:
        """
        Analyze research findings to extract key insights and patterns.
        
        This method uses the LLM to analyze the search results and extract
        key concepts, technical depth assessment, trends, and research gaps.
        
        Parameters:
            input_data: Dictionary containing 'findings' (search results)
            
        Returns:
            Dictionary with analysis results including key concepts and trends
        """
        try:
            findings = input_data.get("findings", {})
            prompt = ChatPromptTemplate.from_template("""
            Analyze these research findings and extract key insights:
            {findings}
            
            Return a JSON object with this exact structure:
            {{
                "key_concepts": ["concept1", "concept2", ...],
                "analysis": "detailed analysis of findings",
                "technical_depth": "assessment of technical depth",
                "trends": ["trend1", "trend2", ...],
                "gaps": ["gap1", "gap2", ...]
            }}
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"findings": json.dumps(findings, indent=2)})
            
            # Ensure the result has all required fields
            return {
                "key_concepts": result.get("key_concepts", []),
                "analysis": result.get("analysis", "No analysis available"),
                "technical_depth": result.get("technical_depth", "Not assessed"),
                "trends": result.get("trends", []),
                "gaps": result.get("gaps", [])
            }
            
        except Exception as e:
            print(f"Error analyzing findings: {str(e)}")
            return {
                "key_concepts": [],
                "analysis": f"Error in analysis: {str(e)}",
                "technical_depth": "Not assessed",
                "trends": [],
                "gaps": []
            }

    async def _rank_sources(self, input_data: Dict) -> Dict[str, Any]:
        """
        Rank sources based on multiple criteria with dynamic weighting.
        
        This method evaluates and ranks each source based on several factors:
        relevance to query, source type (academic vs general), recency,
        citation count, and content quality. It uses the LLM to make
        subjective assessments of relevance and quality.
        
        Parameters:
            input_data: Dictionary with 'topic' (query) and 'sources' (search results)
            
        Returns:
            Dictionary with ranked sources and top insights extracted
        """
        try:
            sources = input_data.get("sources", {})
            topic = input_data.get("topic", "")
            
            # Flatten sources from academic and general categories
            combined_sources = []
            for source_type in ["academic", "general"]:
                for source in sources.get(source_type, []):
                    source["source_type"] = source_type  # Ensure source type is marked
                    combined_sources.append(source)
            
            if not combined_sources:
                print("No sources to rank")
                return {"ranked_sources": [], "top_insights": []}
                
            print(f"Ranking {len(combined_sources)} sources")
            
            # Define weights for different ranking factors
            weights = {
                "relevance": 0.35,      # LLM judgment of relevance
                "source_type": 0.20,    # Academic sources preferred over general
                "recency": 0.20,        # More recent sources preferred
                "citation_count": 0.15, # Higher citation counts preferred
                "content_quality": 0.10 # Quality of content assessment
            }
            
            # Short prompt for LLM ranking to avoid token issues
            prompt = ChatPromptTemplate.from_template("""
            Rank these sources for research on: {topic}
            
            For each source, rate (1-10):
            - Relevance: How well it addresses the topic
            - Credibility: Source authority and reliability
            - Technical depth: Level of technical detail
            
            Return a JSON array of objects with:
            - source_idx: index of the source in the input list
            - overall_score: combined score (float, 0-10)
            - relevance: relevance score (float, 0-10)
            - credibility: credibility score (float, 0-10)
            - technical_depth: technical depth score (float, 0-10)
            - key_insights: ["insight1", "insight2", ...] (1-3 key insights)
            
            Example:
            [
              {{"source_idx": 0, "overall_score": 8.5, "relevance": 9, "credibility": 8, "technical_depth": 7, "key_insights": ["insight1"]}},
              {{"source_idx": 1, "overall_score": 7.2, "relevance": 8, "credibility": 7, "technical_depth": 6, "key_insights": ["insight2"]}}
            ]
            """)
            
            # Prepare simplified source representations to save tokens
            simple_sources = []
            for i, source in enumerate(combined_sources):
                simple_sources.append({
                    "idx": i,
                    "title": source.get("title", "Untitled"),
                    "url": source.get("url", ""),
                    "content_snippet": source.get("content", "")[:150],  # Limit content length
                    "source_type": source.get("source_type", "general")
                })
            
            # Get LLM ranking
            chain = prompt | self.llm | JsonOutputParser()
            try:
                llm_ranking = await chain.ainvoke({
                    "topic": topic,
                    "sources": json.dumps(simple_sources)
                })
            except Exception as e:
                print(f"Error in LLM ranking: {str(e)}")
                # Fallback to simpler ranking
                llm_ranking = [{"source_idx": i, "overall_score": 5.0, "key_insights": []} 
                              for i in range(len(combined_sources))]
            
            # Process LLM rankings and apply additional factors
            ranked_results = []
            for rank_info in llm_ranking:
                idx = rank_info.get("source_idx", -1)
                if idx < 0 or idx >= len(combined_sources):
                    continue
                    
                source = combined_sources[idx]
                
                # Start with LLM's overall score normalized to 0-1
                base_score = rank_info.get("overall_score", 5.0) / 10.0
                
                # Apply source type bonus (academic sources get a boost)
                source_type_score = 1.0 if source.get("source_type") == "academic" else 0.7
                
                # Apply recency factor if available
                recency_score = 0.8  # Default
                pub_date = source.get("pub_date", "")
                if pub_date:
                    try:
                        year_match = re.search(r'(\d{4})', pub_date)
                        if year_match:
                            year = int(year_match.group(1))
                            current_year = datetime.now().year
                            # Newer is better: 1.0 for current year, decreasing for older
                            recency_score = max(0.5, 1.0 - (current_year - year) * 0.1)
                    except:
                        pass
                        
                # Apply citation factor if available
                citation_score = 0.7  # Default
                citations = source.get("citation_count", 0)
                if citations > 0:
                    # More citations is better (up to a point)
                    citation_score = min(1.0, 0.7 + (citations / 100) * 0.3)
                
                # Calculate final weighted score
                final_score = (
                    weights["relevance"] * base_score +
                    weights["source_type"] * source_type_score +
                    weights["recency"] * recency_score +
                    weights["citation_count"] * citation_score +
                    weights["content_quality"] * (rank_info.get("technical_depth", 5.0) / 10.0)
                )
                
                # Create enhanced source with ranking info
                enhanced_source = {
                    "title": source.get("title", "Untitled"),
                    "url": source.get("url", ""),
                    "content": source.get("content", ""),
                    "source_type": source.get("source_type", "general"),
                    "pub_date": source.get("pub_date", ""),
                    "citation_count": source.get("citation_count", 0),
                    "overall_score": final_score,
                    "score_breakdown": {
                        "relevance": base_score,
                        "source_type": source_type_score,
                        "recency": recency_score,
                        "citation": citation_score,
                        "technical_depth": rank_info.get("technical_depth", 5.0) / 10.0
                    },
                    "key_insights": rank_info.get("key_insights", [])
                }
                
                ranked_results.append(enhanced_source)
            
            # Sort by overall score
            ranked_results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
            
            return {
                "ranked_sources": ranked_results,
                "top_insights": self._extract_top_insights(ranked_results)
            }
            
        except Exception as e:
            print(f"Error ranking sources: {str(e)}")
            return {"ranked_sources": [], "top_insights": []}

    def _extract_top_insights(self, ranked_sources: List[Dict]) -> List[str]:
        """
        Extract and deduplicate top insights from ranked sources.
        
        This method collects key insights from all ranked sources, removes
        duplicates, and returns the most important insights.
        
        Parameters:
            ranked_sources: List of sources with key_insights attributes
            
        Returns:
            List of unique insights across all sources
        """
        all_insights = []
        for source in ranked_sources:
            insights = source.get("key_insights", [])
            all_insights.extend(insights)
        
        # Deduplicate insights while preserving order
        seen = set()
        unique_insights = []
        for insight in all_insights:
            if insight not in seen:
                seen.add(insight)
                unique_insights.append(insight)
        
        return unique_insights[:10]  # Return top 10 unique insights

    async def _evolve_hypothesis(self, input_data: Dict) -> Dict[str, Any]:
        """
        Evolve research hypothesis based on findings.
        
        This method uses the LLM to refine and evolve the current hypothesis
        based on new information discovered during the research process.
        
        Parameters:
            input_data: Dictionary with 'hypothesis' and 'findings'
            
        Returns:
            Dictionary with refined hypothesis and reasoning
        """
        try:
            current_findings = input_data.get("findings", {})
            prompt = ChatPromptTemplate.from_template("""
            Evolve this hypothesis based on new findings:
            Current hypothesis: {hypothesis}
            Findings: {findings}
            
            Return a JSON object with:
            - refined_hypothesis: Updated hypothesis
            - reasoning: Explanation of changes
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            return await chain.ainvoke({
                "hypothesis": input_data.get("hypothesis", ""),
                "findings": current_findings
            })
            
        except Exception as e:
            print(f"Error evolving hypothesis: {str(e)}")
            return {"error": str(e)}

    async def _check_proximity(self, input_data: Dict) -> Dict[str, Any]:
        """
        Check similarity with past research.
        
        This method compares the current query with past research sessions
        to identify similar topics and potential insights that could be reused.
        
        Parameters:
            input_data: Dictionary with 'query' to check against past research
            
        Returns:
            Dictionary with similar past queries and relevance scores
        """
        try:
            query = input_data.get("query", "")
            prompt = ChatPromptTemplate.from_template("""
            Find similar past research for: {query}
            Past research: {memory}
            
            Return a JSON object with:
            - similar_queries: List of similar past queries
            - relevance_scores: List of relevance scores
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            return await chain.ainvoke({
                "query": query,
                "memory": self.memory_store
            })
            
        except Exception as e:
            print(f"Error checking proximity: {str(e)}")
            return {"error": str(e)}

    async def _deep_research(self, input_data: Dict) -> Dict[str, Any]:
        """
        Perform deep research on specific topics.
        
        This method conducts in-depth research on particular subtopics of interest,
        performing focused searches and detailed analysis for each topic.
        
        Parameters:
            input_data: Dictionary with 'topics' list for deep research
            
        Returns:
            Dictionary with detailed research results for each topic
        """
        try:
            topics = input_data.get("topics", [])
            results = {}
            
            for topic in topics:
                # Perform focused web search
                search_results = await self._web_search({"query": topic})
                
                # Analyze findings
                analysis = await self._analyze_findings({"findings": search_results})
                
                # Rank sources
                ranking = await self._rank_sources({
                    "sources": search_results,
                    "topic": topic
                })
                
                results[topic] = {
                    "search_results": search_results,
                    "analysis": analysis,
                    "ranking": ranking
                }
            
            return results
            
        except Exception as e:
            print(f"Error in deep research: {str(e)}")
            return {"error": str(e)}

    async def _get_feedback(self, input_data: Dict) -> Dict[str, Any]:
        """
        Process user feedback and update research direction.
        
        This method analyzes user feedback to adjust the research approach
        and suggest new areas to explore based on user input.
        
        Parameters:
            input_data: Dictionary with 'feedback' from the user
            
        Returns:
            Dictionary with suggested adjustments and new queries
        """
        try:
            feedback = input_data.get("feedback", "")
            prompt = ChatPromptTemplate.from_template("""
            Process this feedback and suggest research adjustments:
            Feedback: {feedback}
            Current research: {current_research}
            
            Return a JSON object with:
            - adjustments: List of research adjustments
            - new_queries: List of new queries to explore
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            return await chain.ainvoke({
                "feedback": feedback,
                "current_research": self._get_current_context()
            })
            
        except Exception as e:
            print(f"Error processing feedback: {str(e)}")
            return {"error": str(e)}

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current research context."""
        return {
            "memory_store": self.memory_store,
            "current_iteration": len(self.memory_store),
            "recent_findings": self.memory_store[-1] if self.memory_store else None
        }

    def _store_results(self, query: str, results: Dict[str, Any]) -> None:
        """
        Store research results temporarily for the current session.
        
        This method saves results from a single research iteration into the
        current_session_results collection, which will be later combined and
        saved to permanent history at the end of all iterations.
        
        Parameters:
            query: The research query
            results: Results from the current research iteration
        """
        try:
            # Create a compact version of results for storage
            compact_results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),  # Convert datetime to string
                "key_concepts": results.get("reflection", {}).get("key_concepts", []),
                "top_sources": [
                    {
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "score": source.get("overall_score", 0)
                    }
                    for source in results.get("ranking", {}).get("ranked_sources", [])[:3]
                ],
                "summary": results.get("reflection", {}).get("analysis", "")[:300]  # Limit summary size
            }
            
            # Add to the session-level collection
            self.current_session_results.append(compact_results)
            
            print(f"Results from iteration {results.get('cycle', 0)} collected for final storage")
            
        except Exception as e:
            print(f"Error collecting results: {str(e)}")

    def _save_combined_session_results(self, query: str, meta_review: Dict[str, Any]) -> None:
        """
        Save combined results from all iterations to permanent history.
        
        This method takes the results from all iterations in the current session,
        combines them with the meta-review, and saves them to the persistent
        memory store. This happens once at the end of a complete research session.
        
        Parameters:
            query: The research query
            meta_review: The meta-review of all research iterations
        """
        try:
            if not self.current_session_results:
                print("No results to save")
                return
                
            # Combine key concepts across all iterations
            all_key_concepts = []
            for result in self.current_session_results:
                all_key_concepts.extend(result.get("key_concepts", []))
            
            # Remove duplicates while preserving order
            unique_key_concepts = []
            seen = set()
            for concept in all_key_concepts:
                if concept not in seen:
                    seen.add(concept)
                    unique_key_concepts.append(concept)
            
            # Get the top sources across all iterations
            all_sources = []
            for result in self.current_session_results:
                all_sources.extend(result.get("top_sources", []))
            
            # Sort sources by score and take top 3
            top_sources = sorted(all_sources, key=lambda x: x.get("score", 0), reverse=True)[:3]
            
            # Create the consolidated memory item
            memory_item = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "key_concepts": unique_key_concepts[:10],  # Limit to top 10 concepts
                "top_sources": top_sources,
                "summary": meta_review.get("evolution_summary", "")[:500]  # Use meta-review summary
            }
            
            # Add to memory store
            self.memory_store.append(memory_item)
            
            # Maintain history size
            if len(self.memory_store) > MAX_HISTORY:
                self.memory_store = self.memory_store[-MAX_HISTORY:]
            
            # Save to file
            self._save_history()
            
            # Clear the session results for next query
            self.current_session_results = []
            
            print("Combined session results saved to history")
            
        except Exception as e:
            print(f"Error saving combined results: {str(e)}")

    def _incorporate_feedback(self, feedback: str) -> None:
        """
        Incorporate user feedback into the most recent research memory.
        
        This method attaches user feedback to the most recent research 
        session, which can be used to refine future research.
        
        Parameters:
            feedback: User feedback text to incorporate
        """
        try:
            if self.memory_store and len(self.memory_store) > 0:
                # Memory store items are dictionaries, not objects
                if isinstance(self.memory_store[-1], dict):
                    self.memory_store[-1]['feedback'] = feedback
                else:
                    # Fall back to attribute setting if it's an object
                    self.memory_store[-1].feedback = feedback
                
                print(f"Feedback successfully incorporated: '{feedback[:50]}...'")
            else:
                print("No previous research session found to incorporate feedback.")
        except Exception as e:
            print(f"Error incorporating feedback: {str(e)}")

    async def _generate_meta_review(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate final meta-review of research with token optimization.
        
        This method creates a comprehensive synthesis of all research iterations,
        identifying consistent findings, best solutions, and suggested next steps.
        It provides an overview of how understanding evolved during the research.
        
        Parameters:
            results: List of iteration results to synthesize
            
        Returns:
            Dictionary with meta-review including evolution summary and key findings
        """
        try:
            # Create a more token-efficient version of results for the meta-review
            compact_results = []
            for result in results:
                # Extract only essential information
                compact_result = {
                    "cycle": result.get("cycle", 0),
                    "hypothesis": result.get("hypothesis", ""),
                    "key_concepts": result.get("reflection", {}).get("key_concepts", []),
                    "analysis_summary": result.get("reflection", {}).get("analysis", "")[:300],  # Limit text length
                    "top_sources": [
                        {
                            "title": s.get("title", ""),
                            "score": s.get("overall_score", 0),
                            "key_insights": s.get("key_insights", [])[:2]  # Limit insights
                        }
                        for s in result.get("ranking", {}).get("ranked_sources", [])[:3]  # Only top 3 sources
                    ]
                }
                compact_results.append(compact_result)
            
            # Construct a concise prompt for meta-review
            prompt = ChatPromptTemplate.from_template("""
            Generate a concise meta-review of these research iterations:
            {results}
            
            Return a JSON object with:
            {{
                "evolution_summary": "Brief explanation of how understanding evolved across iterations (max 250 words)",
                "consistent_findings": ["finding1", "finding2", ...],
                "best_solutions": ["solution1", "solution2", ...],
                "next_steps": ["step1", "step2", ...],
                "research_quality": {{
                    "coverage": "brief assessment",
                    "depth": "brief assessment",
                    "gaps": ["gap1", "gap2", ...]
                }}
            }}
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({"results": json.dumps(compact_results, indent=2)})
            
            return result
            
        except Exception as e:
            print(f"Error generating meta-review: {str(e)}")
            # Create fallback result with error information
            return {
                "evolution_summary": f"Error in meta-review: {str(e)}",
                "consistent_findings": ["Unable to generate consistent findings due to an error"],
                "best_solutions": ["Error occurred during meta-review generation"],
                "next_steps": ["Try again with a more focused research query"],
                "research_quality": {
                    "coverage": "Not assessed due to error",
                    "depth": "Not assessed due to error",
                    "gaps": ["Assessment incomplete due to error"]
                }
            }

    def _load_history(self) -> None:
        """
        Load research history from file.
        
        This method reads previously saved research sessions from the history
        file and populates the memory_store with them. If the file doesn't
        exist or is invalid, it initializes an empty memory store.
        """
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    self.memory_store = json.load(f)
        except Exception as e:
            print(f"Error loading history: {str(e)}")
            self.memory_store = []

    def _save_history(self) -> None:
        """
        Save research history to file.
        
        This method saves the current memory_store to the history file so
        that research sessions persist between program restarts.
        """
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.memory_store, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {str(e)}")

    def reset(self) -> None:
        """
        Reset the research agent and clear memory.
        
        This method clears the memory store and saves the empty state to
        the history file, essentially resetting the agent to its initial state.
        Use this when starting a completely new research topic or when
        previous research is no longer relevant.
        """
        self.memory_store = []
        self._save_history()
        print("Research agent has been reset. Ready for new research topic.")

    def _format_iteration_output(self, iteration: int, results: Dict) -> str:
        """Format iteration results for display."""
        output = []
        output.append(f"\n========================================\n=== Iteration {iteration} ===\n========================================\n")
        
        # Search queries
        output.append("Search Queries Used:")
        output.append(f"Main Query: {results.get('query', 'N/A')}")
        output.append("Technical Queries:")
        for q in results.get("technical_queries", []):
            output.append(f"- {q}")
        output.append("Trend Queries:")
        for q in results.get("trend_queries", []):
            output.append(f"- {q}")
        output.append("")
        
        # Report
        output.append("Report:")
        output.append(results.get("report", "No report available"))
        output.append("")
        
        # Key concepts
        output.append("Key Concepts:")
        for concept in results.get("key_concepts", []):
            output.append(f"   - {concept}")
        output.append("")
        
        # Top references with detailed scores
        output.append("Top References:")
        ranked_sources = results.get("ranking", {}).get("ranked_sources", [])
        if ranked_sources:
            for idx, source in enumerate(ranked_sources[:5], 1):
                output.append(f"{idx}. {source.get('title', 'Untitled')}")
                output.append(f"   Score: {source.get('overall_score', 0.0):.2f}")
                
                # Display score breakdown if available
                if source.get("score_breakdown"):
                    breakdown = source["score_breakdown"]
                    output.append(f"   Score Breakdown:")
                    for factor, score in breakdown.items():
                        output.append(f"     - {factor.capitalize()}: {score:.2f}")
                
                output.append(f"   Type: {source.get('source_type', 'general').capitalize()}")
                output.append(f"   URL: {source.get('url', 'N/A')}")
                
                # Display key insights if available
                if source.get("key_insights"):
                    output.append(f"   Key Insights:")
                    for insight in source["key_insights"][:2]:  # Show top 2 insights
                        output.append(f"     - {insight}")
                output.append("")
        else:
            output.append("  No ranked sources available for this iteration")
        
        return "\n".join(output)

    def _format_meta_review(self, meta_review: Dict) -> str:
        """Format meta-review for display."""
        output = []
        output.append("\n========================================\n=== FINAL META-REVIEW ===\n========================================\n")
        
        # Evolution summary
        output.append("Evolution Summary:")
        output.append(meta_review.get("evolution_summary", "No evolution summary available"))
        output.append("")
        
        # Consistent findings
        output.append("Consistent Findings:")
        for finding in meta_review.get("consistent_findings", []):
            if isinstance(finding, dict):
                output.append(f"   - {finding.get('finding', 'N/A')}")
            else:
                output.append(f"   - {finding}")
        output.append("")
        
        # Best solutions
        output.append("Best Solutions:")
        for solution in meta_review.get("best_solutions", []):
            if isinstance(solution, dict):
                output.append(f"   - {solution.get('solution', 'N/A')}")
            else:
                output.append(f"   - {solution}")
        output.append("")
        
        # Next steps
        output.append("Next Steps:")
        for step in meta_review.get("next_steps", []):
            if isinstance(step, dict):
                output.append(f"   - {step.get('step', 'N/A')}")
            else:
                output.append(f"   - {step}")
        
        return "\n".join(output)

    async def _perform_deep_research(self, query: str, topics: List[str]) -> Dict[str, Any]:
        """Perform in-depth research on specific topics with advanced reasoning."""
        try:
            print(f"Starting deep research on {len(topics)} topics related to {query}")
            
            deep_results = {}
            for topic in topics:
                print(f"\nResearching topic: {topic}")
                
                # Perform specialized search for this topic
                search_results = await self._web_search({
                    "query": f"{topic} {query} research paper methodology"
                })
                
                # Analyze the findings specifically for this topic
                analysis = await self._analyze_findings({
                    "findings": search_results
                })
                
                # Rank sources for this specific topic
                ranking = await self._rank_sources({
                    "sources": search_results,
                    "topic": topic
                })
                
                # Generate specialized insights for this topic
                insights_prompt = ChatPromptTemplate.from_template("""
                Generate specialized insights for deep research on:
                Topic: {topic} (related to {query})
                
                Research findings:
                {findings}
                
                Top sources:
                {sources}
                
                Return a JSON object with:
                {{
                    "specialized_insights": [
                        {{
                            "insight": "specific technical insight",
                            "explanation": "detailed explanation",
                            "applications": ["application1", "application2"],
                            "connections": ["connection to main query"]
                        }}
                    ],
                    "technical_details": "deep technical explanation of the topic",
                    "integration": "how this topic integrates with {query}",
                    "future_directions": ["direction1", "direction2"]
                }}
                """)
                
                # Create simplified data for the insights prompt
                simplified_data = {
                    "topic": topic,
                    "query": query,
                    "findings": {
                        "key_concepts": analysis.get("key_concepts", []),
                        "analysis_summary": analysis.get("analysis", "")[:300]
                    },
                    "sources": [
                        {"title": s.get("title", ""), "key_insights": s.get("key_insights", [])}
                        for s in ranking.get("ranked_sources", [])[:3]
                    ]
                }
                
                # Get specialized insights
                insights_chain = insights_prompt | self.llm | JsonOutputParser()
                try:
                    specialized_insights = await insights_chain.ainvoke(simplified_data)
                except Exception as e:
                    print(f"Error generating specialized insights: {str(e)}")
                    specialized_insights = {
                        "specialized_insights": [{"insight": "Error generating insights", "explanation": str(e)}],
                        "technical_details": "Unable to generate due to error",
                        "integration": "Unable to determine integration",
                        "future_directions": ["Unable to determine future directions"]
                    }
                
                # Store comprehensive results for this topic
                deep_results[topic] = {
                    "research_data": {
                        "search_results": search_results,
                        "analysis": analysis,
                        "ranking": ranking
                    },
                    "specialized_insights": specialized_insights
                }
                
                print(f"Completed deep research on topic: {topic}")
            
            # Generate synthesis across all deep research topics
            synthesis_prompt = ChatPromptTemplate.from_template("""
            Synthesize findings across these deep research topics:
            Topics: {topics}
            Main query: {query}
            
            Return a JSON object with:
            {{
                "cross_topic_insights": [
                    "insight connecting multiple topics"
                ],
                "synthesis_summary": "comprehensive synthesis across topics",
                "implications_for_main_query": "how these topics together impact the main query",
                "recommendations": [
                    {{
                        "recommendation": "specific recommendation",
                        "supporting_topics": ["topic1", "topic2"]
                    }}
                ]
            }}
            """)
            
            # Generate topic summaries for synthesis
            topic_summaries = {}
            for topic, data in deep_results.items():
                insights = data.get("specialized_insights", {})
                
                # Create a concise summary of the topic's specialized insights
                topic_summaries[topic] = {
                    "insights": [i.get("insight") for i in insights.get("specialized_insights", [])],
                    "integration": insights.get("integration", ""),
                    "future_directions": insights.get("future_directions", [])
                }
            
            # Generate synthesis
            synthesis_chain = synthesis_prompt | self.llm | JsonOutputParser()
            try:
                synthesis = await synthesis_chain.ainvoke({
                    "topics": list(topics),
                    "query": query,
                    "topic_summaries": topic_summaries
                })
            except Exception as e:
                print(f"Error generating synthesis: {str(e)}")
                synthesis = {
                    "cross_topic_insights": ["Error generating cross-topic insights"],
                    "synthesis_summary": f"Error in synthesis: {str(e)}",
                    "implications_for_main_query": "Unable to determine implications",
                    "recommendations": []
                }
            
            # Format final results
            final_results = {
                "deep_research": {
                    "main_query": query,
                    "topics": topics,
                    "topic_results": deep_results,
                    "synthesis": synthesis
                }
            }
            
            print("\nDeep research completed successfully")
            return final_results
            
        except Exception as e:
            print(f"Error in deep research: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Deep research failed: {str(e)}",
                "partial_results": deep_results if locals().get("deep_results") else {}
            }

    async def _refine_query_with_feedback(self, original_query: str, feedback: str) -> Dict[str, Any]:
        """
        Refine the research query based on user feedback.
        
        This method uses the LLM to analyze user feedback and create an improved
        version of the original query that incorporates the feedback. It also
        identifies specific focus areas based on the feedback.
        
        Parameters:
            original_query: The initial research query
            feedback: User feedback to incorporate
            
        Returns:
            Dictionary with refined query, focus areas, and reasoning
        """
        try:
            # Create a prompt to refine the query based on feedback
            refine_prompt = ChatPromptTemplate.from_template("""
            Original research query: {query}
            
            User feedback: {feedback}
            
            Based on this feedback, please refine the research approach.
            
            Return a JSON object with:
            {{
                "refined_query": "improved version of the query that incorporates feedback",
                "focus_areas": ["specific area to focus on", "another area", ...],
                "reasoning": "explanation of how feedback was incorporated"
            }}
            """)
            
            # Process the feedback using the LLM
            chain = refine_prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({
                "query": original_query,
                "feedback": feedback
            })
            
            return result
            
        except Exception as e:
            print(f"Error refining query with feedback: {str(e)}")
            # Return original query if refinement fails
            return {
                "refined_query": original_query,
                "focus_areas": [],
                "reasoning": f"Failed to refine due to error: {str(e)}"
            }

async def main():
    supervisor = SupervisorAgent()
    print("Research Supervisor Initialized. Type 'exit' to end.")
    
    while True:
        try:
            user_input = input("\nEnter your research query: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            result = await supervisor.process_query(user_input)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
                continue
                
            # Display results in detail with improved formatting
            display_research_results(result)
            
            # Get user choice
            choice = input("\nChoose an option:\n1. Provide feedback\n2. Request deep research\n3. Reset and start new topic\n4. Exit\nYour choice: ")
            
            if choice == "1":
                feedback = input("Enter your feedback to improve the research: ")
                if feedback.strip():
                    print("\nRestarting research with your feedback...\n")
                    # Process the query again with feedback
                    feedback_result = await supervisor.process_query(user_input, feedback=feedback)
                    
                    if "error" in feedback_result:
                        print(f"\nError during feedback-based research: {feedback_result['error']}")
                    else:
                        print("\n=== RESEARCH RESULTS BASED ON YOUR FEEDBACK ===\n")
                        display_research_results(feedback_result)
                else:
                    print("No feedback provided. Continuing with current results.")
                
            elif choice == "2":
                topics_input = input("Enter topics for deep research (comma-separated): ")
                topics = [topic.strip() for topic in topics_input.split(",") if topic.strip()]
                if topics:
                    print("\nStarting deep research on specified topics...\n")
                    deep_result = await supervisor.process_query(user_input, deep_research=topics)
                    
                    if "error" in deep_result:
                        print(f"\nError during deep research: {deep_result['error']}")
                    else:
                        display_deep_research_results(deep_result)
                else:
                    print("No valid topics provided")
                
            elif choice == "3":
                supervisor.reset()
                continue
                
            elif choice == "4":
                break
                
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()

def display_research_results(result):
    """Display formatted research results."""
    for cycle in result["cycles"]:
        print(f"\n{'='*50}")
        print(f"=== RESEARCH ITERATION {cycle['cycle']} ===")
        print(f"{'='*50}\n")
        
        # Display query
        print("RESEARCH QUERY:")
        print(f"  {cycle['hypothesis']}")
        
        # Display analysis
        print("\nANALYSIS:")
        print(cycle["reflection"]["analysis"])
        
        # Display key concepts
        print("\nKEY CONCEPTS:")
        for concept in cycle["reflection"]["key_concepts"]:
            print(f"  • {concept}")
        
        # Display trends if available
        if cycle["reflection"].get("trends"):
            print("\nEMERGING TRENDS:")
            for trend in cycle["reflection"]["trends"]:
                print(f"  • {trend}")
        
        # Display gaps if available
        if cycle["reflection"].get("gaps"):
            print("\nRESEARCH GAPS:")
            for gap in cycle["reflection"]["gaps"]:
                print(f"  • {gap}")
        
        # Display top references with detailed scores
        print("\nTOP REFERENCES:")
        ranked_sources = cycle["ranking"].get("ranked_sources", [])
        if ranked_sources:
            for idx, source in enumerate(ranked_sources[:5], 1):
                print(f"{idx}. {source.get('title', 'Untitled')}")
                print(f"   Score: {source.get('overall_score', 0.0):.2f}")
                
                # Display score breakdown if available
                if source.get("score_breakdown"):
                    breakdown = source["score_breakdown"]
                    print(f"   Score Breakdown:")
                    for factor, score in breakdown.items():
                        print(f"     - {factor.capitalize()}: {score:.2f}")
                
                print(f"   Type: {source.get('source_type', 'general').capitalize()}")
                print(f"   URL: {source.get('url', 'N/A')}")
                
                # Display key insights if available
                if source.get("key_insights"):
                    print(f"   Key Insights:")
                    for insight in source["key_insights"][:2]:  # Show top 2 insights
                        print(f"     - {insight}")
                print()
        else:
            print("  No ranked sources available for this iteration")
    
    # Display the meta-review
    print(f"\n{'='*50}")
    print(f"=== FINAL META-REVIEW ===")
    print(f"{'='*50}\n")
    
    # Evolution summary
    print("EVOLUTION SUMMARY:")
    print(result["meta_review"].get("evolution_summary", "No evolution summary available"))
    print()
    
    # Consistent findings
    print("CONSISTENT FINDINGS:")
    for finding in result["meta_review"].get("consistent_findings", []):
        if isinstance(finding, dict):
            print(f"  • {finding.get('finding', 'N/A')}")
            if finding.get("evidence"):
                print(f"    Evidence: {', '.join(finding['evidence'])}")
            if finding.get("confidence"):
                print(f"    Confidence: {finding['confidence']}")
        else:
            print(f"  • {finding}")
    print()
    
    # Best solutions
    print("BEST SOLUTIONS:")
    for solution in result["meta_review"].get("best_solutions", []):
        if isinstance(solution, dict):
            print(f"  • {solution.get('solution', 'N/A')}")
            if solution.get("pros"):
                print(f"    Pros: {', '.join(solution['pros'])}")
            if solution.get("cons"):
                print(f"    Cons: {', '.join(solution['cons'])}")
            if solution.get("feasibility"):
                print(f"    Feasibility: {solution['feasibility']}")
        else:
            print(f"  • {solution}")
    print()
    
    # Next steps
    print("RECOMMENDED NEXT STEPS:")
    for step in result["meta_review"].get("next_steps", []):
        if isinstance(step, dict):
            print(f"  • {step.get('step', 'N/A')}")
            if step.get("priority"):
                print(f"    Priority: {step['priority']}")
            if step.get("rationale"):
                print(f"    Rationale: {step['rationale']}")
        else:
            print(f"  • {step}")

def display_deep_research_results(result):
    """Display formatted deep research results."""
    if "deep_research" not in result:
        print("No deep research results available.")
        return
    
    deep_research = result["deep_research"]
    
    print(f"\n{'='*50}")
    print(f"=== DEEP RESEARCH RESULTS ===")
    print(f"{'='*50}\n")
    
    print(f"Main Query: {deep_research.get('main_query', 'N/A')}")
    print(f"Topics Researched: {', '.join(deep_research.get('topics', []))}")
    
    # Display individual topic results
    for topic, data in deep_research.get("topic_results", {}).items():
        print(f"\n{'='*40}")
        print(f"=== TOPIC: {topic} ===")
        print(f"{'='*40}\n")
        
        # Show specialized insights
        insights = data.get("specialized_insights", {})
        
        print("SPECIALIZED INSIGHTS:")
        for insight in insights.get("specialized_insights", []):
            print(f"  • {insight.get('insight', 'N/A')}")
            print(f"    Explanation: {insight.get('explanation', 'N/A')}")
            if insight.get("applications"):
                print(f"    Applications: {', '.join(insight['applications'])}")
            if insight.get("connections"):
                print(f"    Connections: {', '.join(insight['connections'])}")
            print()
        
        print("TECHNICAL DETAILS:")
        print(f"  {insights.get('technical_details', 'Not available')}")
        
        print("\nINTEGRATION WITH MAIN QUERY:")
        print(f"  {insights.get('integration', 'Not available')}")
        
        print("\nFUTURE DIRECTIONS:")
        for direction in insights.get("future_directions", []):
            print(f"  • {direction}")
        
        # Display top sources for this topic
        print("\nTOP SOURCES:")
        ranked_sources = data.get("research_data", {}).get("ranking", {}).get("ranked_sources", [])
        if ranked_sources:
            for idx, source in enumerate(ranked_sources[:3], 1):
                print(f"{idx}. {source.get('title', 'Untitled')}")
                print(f"   Score: {source.get('overall_score', 0.0):.2f}")
                print(f"   URL: {source.get('url', 'N/A')}")
                print()
    
    # Display synthesis
    synthesis = deep_research.get("synthesis", {})
    
    print(f"\n{'='*50}")
    print(f"=== CROSS-TOPIC SYNTHESIS ===")
    print(f"{'='*50}\n")
    
    print("SYNTHESIS SUMMARY:")
    print(synthesis.get("synthesis_summary", "Not available"))
    
    print("\nCROSS-TOPIC INSIGHTS:")
    for insight in synthesis.get("cross_topic_insights", []):
        print(f"  • {insight}")
    
    print("\nIMPLICATIONS FOR MAIN QUERY:")
    print(synthesis.get("implications_for_main_query", "Not available"))
    
    print("\nRECOMMENDATIONS:")
    for rec in synthesis.get("recommendations", []):
        if isinstance(rec, dict):
            print(f"  • {rec.get('recommendation', 'N/A')}")
            if rec.get("supporting_topics"):
                print(f"    Supporting Topics: {', '.join(rec['supporting_topics'])}")
        else:
            print(f"  • {rec}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())