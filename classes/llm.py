# classes/llm.py
import abc

class LLM(abc.ABC):
    """Abstract Base Class for Language Models."""

    def __init__(self, name: str, api_key: str, model: str, max_tokens: int, temperature: float):
        """
        Initializes the base LLM attributes.

        Args:
            name (str): The name of the LLM provider (e.g., "Claude", "OpenAI").
            api_key (str): The API key for the service.
            model (str): The specific model identifier.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature for generation.
        """
        if not api_key:
            # Immediately raise error if API key is missing during instantiation
            raise ValueError(f"{name} API key is missing.")

        self.name = name
        self.api_key = api_key
        self.model = model
        # Use the provided max_tokens and temperature, not hardcoded values
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        print(f"LLM Base Class initialized for {self.name} with model {self.model}") # Debug print

    @abc.abstractmethod
    def setup_client(self):
        """Abstract method to set up the specific API client."""
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Abstract method to generate text based on a prompt."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
