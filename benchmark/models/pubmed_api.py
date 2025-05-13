"""
PubMed API client for study search.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Union
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import requests

from .. import config
from . import ModelInterface

class PubMedAPI(ModelInterface):
    """PubMed API client for searching clinical trials."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        tool: str = "clinical_trial_crawler"
    ):
        """
        Initialize the PubMed API client.
        
        Args:
            email: Email address for NCBI API (optional, but recommended)
            api_key: NCBI API key (optional, increases rate limit from 3 to 10 requests/second)
            tool: Name of the tool using the API (optional)
        """
        super().__init__("pubmed_api")
        
        # Try to get API key from environment variables if not provided
        if api_key is None:
            api_key = os.environ.get("PUBMED_API_KEY")
        
        # Email is no longer required, but we'll still use it if provided
        self.email = email
        self.api_key = api_key
        self.tool = tool
    
    def search(self, query: str, max_results: int = 20, db: str = "pubmed", return_ids_only: bool = True) -> List[Dict[str, Any]]:
        """
        Search PubMed for the given query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            db: Database to search (pubmed, pmc, etc.)
            
        Returns:
            List of search results with metadata
        """
        # First, search for IDs
        ids = self._search_ids(query, max_results, db)
        
        if not ids:
            return []
        
        # Then fetch details for those IDs
        if return_ids_only:
            return [{"id": id} for id in ids]
        else:
            return self._fetch_details(ids, db)
    
    def _search_ids(self, query: str, max_results: int, db: str) -> List[str]:
        """
        Search for IDs matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            db: Database to search
            
        Returns:
            List of IDs
        """
        # Construct the search URL
        url = f"{self.BASE_URL}/esearch.fcgi"
        
        # Clean and encode the query
        encoded_query = quote_plus(query)
        
        # Prepare parameters
        params = {
            "db": db,
            "term": query,
            "retmax": max_results,
            "usehistory": "y",
            "retmode": "json",
            "sort": "relevance",
        }
        
        # Add optional parameters if provided
        if self.tool:
            params["tool"] = self.tool
        
        if self.email:
            params["email"] = self.email
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Make the request with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                return data.get("esearchresult", {}).get("idlist", [])
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error searching PubMed. Retrying in {retry_delay} seconds... Error: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to search PubMed after {max_retries} attempts: {e}")
                    return []
    
    def _fetch_details(self, ids: List[str], db: str) -> List[Dict[str, Any]]:
        """
        Fetch details for the given IDs.
        
        Args:
            ids: List of IDs to fetch
            db: Database the IDs belong to
            
        Returns:
            List of article details
        """
        # Construct the fetch URL
        url = f"{self.BASE_URL}/esummary.fcgi"
        
        # Prepare parameters
        params = {
            "db": db,
            "id": ",".join(ids),
            "retmode": "json",
        }
        
        # Add optional parameters if provided
        if self.tool:
            params["tool"] = self.tool
        
        if self.email:
            params["email"] = self.email
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Make the request with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                
                if "result" not in data:
                    return []
                
                # Extract article details
                results = []
                for article_id, article_data in data["result"].items():
                    if article_id == "uids":
                        continue
                    
                    # Extract and format relevant fields
                    if isinstance(article_data, dict):
                        article = {
                            "id": article_id,
                            "title": article_data.get("title", ""),
                            "authors": [author.get("name", "") for author in article_data.get("authors", [])],
                            "journal": article_data.get("fulljournalname", ""),
                            "publication_date": article_data.get("pubdate", ""),
                            "doi": article_data.get("doi", ""),
                            "abstract": article_data.get("abstract", ""),
                            "keywords": article_data.get("keywords", []),
                            "article_type": article_data.get("articletype", []),
                            "source": "PubMed",
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                        }
                        results.append(article)
                
                return results
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error fetching details from PubMed. Retrying in {retry_delay} seconds... Error: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to fetch details from PubMed after {max_retries} attempts: {e}")
                    return []
    
    def predict(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Make a prediction for the given input.
        
        For PubMed, this is equivalent to performing a search.
        
        Args:
            input_data: Search query as a string or a dict with a "query" field
            **kwargs: Additional arguments for the search
            
        Returns:
            List of search results
        """
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            max_results = input_data.get("max_results", 20)
        else:
            query = input_data
            max_results = kwargs.get("max_results", 20)
        
        return self.search(query, max_results)
    
    @classmethod
    def from_config(cls, pubmed_config: config.PubMedConfig):
        """
        Create an instance from a configuration object.
        
        Args:
            pubmed_config: PubMed API configuration
            
        Returns:
            PubMedAPI instance
        """
        return cls(
            email=pubmed_config.email,
            api_key=pubmed_config.api_key,
            tool=pubmed_config.tool
        ) 