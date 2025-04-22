import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from functools import lru_cache

# You'll need to install: pip install tavily-python
from tavily import TavilyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    """Container for research results with useful metadata"""
    query: str
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = None
    raw_response: Dict[str, Any] = None
    status: str = "success"
    error: Optional[str] = None

# Global research tool instance
_research_tool = None

def _get_research_tool():
    """
    Get or initialize the TavilyClient instance.
    
    Returns:
        TavilyClient: Initialized Tavily client
    """
    global _research_tool
    if _research_tool is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable must be set")
        _research_tool = TavilyClient(api_key)
    return _research_tool

def research(query: str, include_sources: bool = True) -> Dict[str, Any]:
    """
    Perform research using Tavily API.
    
    This function is designed to be used as a tool for agents in the autogen framework.
    
    Args:
        query: The research query
        include_sources: Whether to include source information in the result
        
    Returns:
        Dictionary containing research results with answer and sources if requested
    """
    try:
        client = _get_research_tool()
        response = client.search(
            query=query,
            include_answer="basic"
        )
        
        result = ResearchResult(
            query=query,
            answer=response.get("answer"),
            sources=response.get("results", []),
            raw_response=response,
            status="success"
        )
        
        response_dict = {
            "answer": result.answer or "No answer found",
            "status": result.status
        }
        
        if include_sources and result.sources:
            sources = [
                {
                    "title": source.get("title", "No title"),
                    "url": source.get("url", ""),
                    "snippet": source.get("content", "No content available")
                }
                for source in result.sources[:5]  # Limit to top 5 sources
            ]
            response_dict["sources"] = sources
        
        return response_dict
    
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        return {
            "status": "error",
            "error": f"Research failed: {str(e)}",
            "answer": None
        }

def perform_multi_research(queries: List[str], include_sources: bool = False) -> List[Dict[str, Any]]:
    """
    Perform multiple research queries using Tavily API.
    
    Args:
        queries: List of research queries
        include_sources: Whether to include source information in results
        
    Returns:
        List of dictionaries containing research results
    """
    results = []
    
    for query in queries:
        try:
            result = research(query, include_sources)
            result["query"] = query  # Add the query to the result
            results.append(result)
        except Exception as e:
            logger.error(f"Research failed for query '{query}': {str(e)}")
            results.append({
                "query": query,
                "status": "error",
                "error": f"Research failed: {str(e)}",
                "answer": None
            })
    
    return results