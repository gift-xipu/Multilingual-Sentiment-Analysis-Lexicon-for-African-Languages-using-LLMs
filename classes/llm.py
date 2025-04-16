class LLM():
    def __init__ (self, name, api_key, model, max_tokens, temperature):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.max_tokens = 1000
        self.temperature = 0
        self.client = None

