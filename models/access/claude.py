# models/access/claude.py
import anthropic
import traceback
from classes.llm import LLM

# Claude-specific implementation
class ClaudeLLM(LLM):
    # Default to a known recent model, allow override
    DEFAULT_MODEL = "claude-3-5-sonnet-20240620"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 4096):
        """Initializes the Claude LLM."""
        print(f"Initializing ClaudeLLM with API Key: {'*' * 5}{api_key[-4:] if api_key else 'None'}") # Debug print
        super().__init__("Claude", api_key, model, max_tokens, temperature)
        # self.client is initialized in setup_client

    def setup_client(self):
        """Initializes the Anthropic client."""
        if self.client:
            print("Anthropic client already initialized.")
            return self.client

        print("Setting up Anthropic client...")
        # This is the standard way to initialize the client for v1.0+.
        # It does NOT take a 'proxies' argument directly here.
        # Proxy configuration is typically handled by environment variables
        # (HTTPS_PROXY, HTTP_PROXY) which the underlying httpx library uses.
        try:
            # Ensure API key is available before creating client
            if not self.api_key:
                raise ValueError("Cannot setup Claude client: API key is missing.")

            # Use anthropic.Anthropic for v1+ library
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("Anthropic client initialized successfully.") # Debug print
            return self.client
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            print(f"Full traceback during client setup:\n{traceback.format_exc()}")
            self.client = None # Ensure client is None if setup fails
            # Re-raise the exception so initialize_llm can catch it
            raise

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates text using the Anthropic Messages API."""
        # Set up client if not already done
        if not self.client:
            print("Client not initialized. Setting up client in generate()...") # Debug print
            try:
                self.setup_client()
            except Exception as e:
                 # If setup fails here, we cannot proceed
                 error_message = f"Anthropic client setup failed: {e}. Cannot generate text."
                 print(error_message)
                 # Return an error message or raise a specific exception for the UI
                 return f"ERROR: {error_message}"


        # Check again if client setup failed
        if not self.client:
             error_message = "Anthropic client is not available after setup attempt. Cannot generate text."
             print(error_message)
             return f"ERROR: {error_message}"


        print(f"Generating text with Claude model: {self.model}") # Debug print
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
            print("Using system prompt.") # Debug print

        try:
            # Send request to Claude using the Messages API
            print("Sending request to Anthropic API...") # Debug print
            response = self.client.messages.create(**message_params)
            print("Received response from Anthropic API.") # Debug print

            # Return the text response
            # Check if content is present and has text
            if response.content and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                return response.content[0].text
            else:
                # Handle cases where the response might be structured differently or empty
                error_message = f"Unexpected response structure or empty content: {response}"
                print(f"Warning: {error_message}")
                return f"ERROR: {error_message}"
        except Exception as e:
            error_message = f"Error during Anthropic API call: {e}"
            print(error_message)
            print(f"Full traceback during API call:\n{traceback.format_exc()}")
            # Return an error message for the UI
            return f"ERROR: {error_message}"
