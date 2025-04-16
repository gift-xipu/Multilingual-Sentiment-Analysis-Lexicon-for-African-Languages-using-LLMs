import anthropic
from classes.llm import LLM

# Claude-specific implementation
class ClaudeLLM(LLM):
    def __init__(self, api_key, model="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=4096):
        super().__init__("Claude", api_key, model, max_tokens, temperature)
    
    def setup_client(self):
        self.client = anthropic.Client(api_key=self.api_key)
        return self.client
    
    def generate(self, prompt, system_prompt=None):
        # Set up client if not already done
        if not self.client:
            self.setup_client()
        
        # Prepare the message parameters
        message_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Send request to Claude
        response = self.client.messages.create(**message_params)
        
        # Return the text response
        return response.content[0].text
