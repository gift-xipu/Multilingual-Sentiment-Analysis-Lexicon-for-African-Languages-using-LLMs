
# models/access/groq.py (or similar location)

import os
from groq import Groq
from classes.llm import LLM # Assuming your base LLM class is here

# Groq-specific implementation
class LlamaLLM(LLM):
    def __init__(self, api_key, model="llama3-70b-8192", temperature=0.0, max_tokens=4096):
        # Note: Changed default model to a known Groq model.
        # Replace "llama3-70b-8192" with your target model ID when available.
        super().__init__("Groq", api_key, model, max_tokens, temperature)
        # Ensure API key is set (consider handling None case better if needed)
        if not api_key:
             raise ValueError("Groq API key is required but was not provided or found in environment variables.")
        self.api_key = api_key


    def setup_client(self):
        """Initializes the Groq API client."""
        # The client automatically uses the api_key provided during instantiation
        self.client = Groq(api_key=self.api_key)
        # Optional: Add timeout or other configurations here if needed
        # self.client = Groq(api_key=self.api_key, timeout=60.0)
        return self.client

    def generate(self, prompt, system_prompt=None):
        """Generates text using the configured Groq model."""
        # Set up client if not already done
        if not self.client:
            self.setup_client()

        messages = []
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        try:
            # Send request to Groq API (uses OpenAI's chat completions format)
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # Add other parameters like top_p if needed
                # top_p=1,
                # stop=None,
                # stream=False,
            )

            # Return the text response
            return chat_completion.choices[0].message.content

        except Exception as e:
            # Handle potential API errors (e.g., rate limits, invalid key)
            print(f"Error generating response from Groq: {e}")
            # Re-raise or return an error message/None depending on desired behavior
            raise # Or return None / f"Error: {e}"
