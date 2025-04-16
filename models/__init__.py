# Import model classes directly from the models directory
from models.access.claude import ClaudeLLM
from models.access.openai import OpenAILLM
from models.access.gemini import GeminiLLM
from models.access.llama import LlamaLLM # <-- Import GroqLLM

# Function to initialize all LLMs
def initialize_llms(claude_api_key=None, openai_api_key=None, gemini_api_key=None, groq_api_key=None): # <-- Add groq_api_key parameter
    """
    Initialize multiple LLM instances with provided API keys.

    Args:
        claude_api_key (str, optional): API key for Claude. Defaults to None.
        openai_api_key (str, optional): API key for OpenAI. Defaults to None.
        gemini_api_key (str, optional): API key for Gemini. Defaults to None.
        groq_api_key (str, optional): API key for Groq. Defaults to None. # <-- Add groq_api_key docstring

    Returns:
        dict: Dictionary containing initialized LLM instances keyed by model provider name (e.g., "claude", "groq").
    """
    llms = {}

    # Initialize Claude if API key is provided
    if claude_api_key:
        try:
            claude = ClaudeLLM(
                api_key=claude_api_key,
                temperature=0.0
            )
            claude.setup_client()
            llms["claude"] = claude
            print("ClaudeLLM initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize ClaudeLLM: {e}")


    # Initialize OpenAI if API key is provided
    if openai_api_key:
        try:
            openai_llm = OpenAILLM(
                api_key=openai_api_key,
                temperature=0.0
            )
            openai_llm.setup_client()
            llms["openai"] = openai_llm
            print("OpenAILLM initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize OpenAILLM: {e}")

    # Initialize Gemini if API key is provided
    if gemini_api_key:
        try:
            gemini = GeminiLLM(
                api_key=gemini_api_key,
                temperature=0.0
            )
            gemini.setup_client()
            llms["gemini"] = gemini
            print("GeminiLLM initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize GeminiLLM: {e}")

    # Initialize Groq if API key is provided
    if groq_api_key: # <-- Add Groq initialization block
        try:
            groq = LlamaLLM(
                api_key=groq_api_key,
                # It will use the default model from GroqLLM class: "meta-llama/llama-4-scout-17b-16e-instruct"
                # Or you could explicitly set it here:
                # model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0
            )
            groq.setup_client() # Ensure the client is ready
            llms["groq"] = groq
            print(f"GroqLLM initialized successfully (requesting model: {groq.model}).")
             # Add the warning about model availability
            print(f"  Note: API calls will fail if model '{groq.model}' is not available on Groq.")
        except Exception as e:
            print(f"Failed to initialize GroqLLM: {e}")


    if not llms:
        print("Warning: No LLMs were initialized. Please provide at least one valid API key.")

    return llms

