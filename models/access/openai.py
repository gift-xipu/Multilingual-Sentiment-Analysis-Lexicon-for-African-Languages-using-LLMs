from openai import OpenAI
from classes.llm import LLM

# OpenAI-specific implementation
class OpenAILLM(LLM):
    def __init__(self, api_key, model="gpt-4o", temperature=0.0, max_tokens=4096):
        # Call the parent constructor with the name "OpenAI"
        super().__init__("OpenAI", api_key, model, max_tokens, temperature)
        self.base_url = None  # Optional base URL for API requests
    
    def setup_client(self):
        """Set up the OpenAI client."""
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        return self.client
    
    def set_base_url(self, base_url):
        """
        Set a custom base URL for API requests (useful for proxies or Azure OpenAI).
        Args:
            base_url (str): The base URL to use for API requests
        """
        self.base_url = base_url
        # Reset client if it was already set up
        if self.client:
            self.setup_client()
        return self
    
    def generate(self, prompt, system_prompt=None):
        """
        Generate a response from OpenAI.
        Args:
            prompt (str): The user prompt
            system_prompt (str, optional): System instructions for the AI
        Returns:
            str: OpenAI's response
        """
        # Set up client if not already done
        if not self.client:
            self.setup_client()
        
        # Prepare the messages list
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Send request to OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Return the text response
        return response.choices[0].message.content
