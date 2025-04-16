import google.generativeai as genai
from classes.llm import LLM

# Gemini-specific implementation
class GeminiLLM(LLM):
    def __init__(self, api_key, model="gemini-1.5-pro", temperature=0.0, max_tokens=4096):
        super().__init__("Gemini", api_key, model, max_tokens, temperature)
    
    def setup_client(self):
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        self.client = genai
        return self.client
    
    def generate(self, prompt, system_prompt=None):
        # Set up client if not already done
        if not self.client:
            self.setup_client()
        
        # Get the model
        model = self.client.GenerativeModel(self.model)
        
        # Create generation config
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": 0.95,
            "top_k": 64
        }
        
        # Prepare the message
        if system_prompt:
            # For Gemini, we prepend the system prompt to the user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Return the text response
        return response.text
