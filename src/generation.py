"""
RAG (Retrieval-Augmented Generation) module for generating responses based on retrieved context.

This module implements a RAG system that retrieves relevant documents and uses them to generate
responses to user queries using a language model.
"""

import torch
from typing import Tuple, List
from .utils import retriver, load_llm_model

class RAG_Generation:
    """
    A class that implements Retrieval-Augmented Generation for question answering.
    """

    def __init__(self) -> None:
        """
        Initialize the RAG generation system.
        """
        self.retriever = retriver.Retrieve()
        self.model, self.tokenizer = load_llm_model.Model()

    def generate_output(self, query: str) -> str:
        """
        Generate a response to the given query using RAG.

        Args:
            query (str): The user's question or query.

        Returns:
            str: The generated response.
        """
        context = self.retriever.semantic_search(query=query)
        prompt = self._create_prompt(context, query)
        messages = self._create_messages(prompt)
        
        input_tokens = self._tokenize_input(messages)
        output_tokens = self._generate_tokens(input_tokens)
        output = self._decode_output(output_tokens = output_tokens, input_tokens_length = len(input_tokens))
        
        return output

    def _create_prompt(self, context: str, query: str) -> str:
        """Create a prompt for the language model."""
        return f"""Answer the question based ONLY on the following context:
                context: {context}
                question: {query}
                """

    def _create_messages(self, prompt: str) -> List[dict]:
        """Create a list of message dictionaries for the language model."""
        return [{"role": "user", "content": prompt}]

    def _tokenize_input(self, messages: List[dict]) -> List[int]:
        """Tokenize the input messages."""
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def _generate_tokens(self, input_tokens: List[int]) -> torch.Tensor:
        """Generate output tokens using the language model."""
        return self.model.generate(torch.tensor([input_tokens]), max_new_tokens=500)

    def _decode_output(self, output_tokens: torch.Tensor, input_tokens_length: int) -> str:
        """Decode the output tokens into a string."""
        return self.tokenizer.decode(output_tokens[0][input_tokens_length:], skip_special_tokens = True)











    
