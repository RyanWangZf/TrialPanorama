"""
Azure OpenAI model interface using Langchain.
"""

import os
import time
from typing import Dict, List, Any, Optional, Union
import json

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .. import config
from . import ModelInterface

class AzureOpenAIModel(ModelInterface):
    """Azure OpenAI model interface."""
    
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: str = "2023-07-01-preview",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_effort: str = "low"
    ):
        """
        Initialize the Azure OpenAI model.
        
        Args:
            model_name: Name of the model deployment in Azure
            api_key: Azure OpenAI API key (will try to get from env if None)
            api_base: Azure OpenAI API base URL (will try to get from env if None)
            api_version: Azure OpenAI API version
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name)
        
        # Try to get from environment variables if not provided
        if api_key is None:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if api_base is None:
            api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if api_key is None or api_base is None:
            raise ValueError("API key and endpoint must be provided in .env file with AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        
        if model_name in ["o3-mini", "o3-preview"]: # temperature is not used for o3-mini
            self.llm = AzureChatOpenAI(
                azure_deployment=model_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
                model_kwargs={"max_completion_tokens": max_tokens, "reasoning_effort": reasoning_effort}
            )
        else:
            self.llm = AzureChatOpenAI(
                azure_deployment=model_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
                temperature=temperature,
                model_kwargs={"max_completion_tokens": max_tokens}
            )

    
    def predict(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        system_prompt: Optional[str] = None,
        output_format: str = "text",
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """
        Make a prediction with the model.
        
        Args:
            input_data: Input data for the model, either a string or a dictionary
            system_prompt: System prompt to use
            output_format: Format of the output, either "text" or "json"
            few_shot_examples: List of few-shot examples to include in the prompt
            **kwargs: Additional arguments to pass to the Langchain chain
            
        Returns:
            Model prediction
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for example in few_shot_examples:
                if "input" in example and "output" in example:
                    messages.append(HumanMessage(content=example["input"]))
                    messages.append(SystemMessage(content=example["output"]))
        
        # Handle input data format
        if isinstance(input_data, str):
            prompt = input_data
        else:
            # Convert dict to a formatted string
            prompt = json.dumps(input_data, indent=2)
        
        # Add the actual input
        messages.append(HumanMessage(content=prompt))
        
        # Create output parser based on format
        if output_format.lower() == "json":
            output_parser = JsonOutputParser()
        else:
            output_parser = StrOutputParser()
        
        # Make prediction with retry logic for rate limiting
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                # Execute the chain
                if output_format.lower() == "json":
                    result = self.llm.invoke(messages, response_format={
                        "type": "json_object"
                    })
                else:
                    result = self.llm.invoke(messages)
                
                # Parse the output
                if output_format.lower() == "json":
                    try:
                        # Try to parse JSON from the content
                        return json.loads(result.content)
                    except json.JSONDecodeError:
                        # If parsing fails, return the raw content
                        return result.content
                else:
                    return result.content
            
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

    @classmethod
    def from_config(cls, model_config: config.ModelConfig, azure_config: config.AzureOpenAIConfig):
        """
        Create an instance from configuration objects.
        
        Args:
            model_config: Model configuration
            azure_config: Azure OpenAI configuration
            
        Returns:
            AzureOpenAIModel instance
        """
        return cls(
            model_name=model_config.model_name,
            api_key=azure_config.api_key,
            api_base=azure_config.api_base,
            api_version=azure_config.api_version,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            reasoning_effort=model_config.reasoning_effort
        ) 