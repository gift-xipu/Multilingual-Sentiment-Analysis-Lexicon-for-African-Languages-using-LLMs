
# utils/shared_ui.py
import streamlit as st
import pandas as pd
import io
import os

# --- Try to import LLMs safely ---
ClaudeLLM, OpenAILLM, GeminiLLM, LlamaLLM = None, None, None, None
_llm_import_error = None  # Save error to warn later

try:
    from models.access.claude import ClaudeLLM
    from models.access.openai import OpenAILLM
    from models.access.gemini import GeminiLLM
    from models.access.llama import LlamaLLM
except ImportError as e:
    _llm_import_error = e


def _llm_import_warning():
    """Show warning if LLM classes failed to import"""
    if _llm_import_error:
        st.warning(f"Could not import all LLM classes ({_llm_import_error}). Check file paths and class names.")


def ensure_session_state():
    """Initializes required session state variables if they don't exist."""
    defaults = {
        "api_keys": {"claude": "", "openai": "", "gemini": "", "groq": ""},
        "current_model_name": "Claude",
        "current_language": "sotho",
        "current_model": None,
        "generated_lexicon": pd.DataFrame(columns=["word", "sentiment", "intensity"]),
        "sentiment_analysis_results": pd.DataFrame(),
        "sentiment_input_text": "",
        "single_analysis_result": None,
        "selected_explanation_item": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_sidebar():
    """Displays sidebar configuration and returns selected model name and language."""
    st.sidebar.header("Model Configuration")
    available_models = ["Claude", "OpenAI", "Gemini", "Llama 4"]
    current_model_index = available_models.index(st.session_state.get("current_model_name", "Claude")) if st.session_state.get("current_model_name", "Claude") in available_models else 0

    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        available_models,
        index=current_model_index,
        key="model_selector_sidebar"
    )
    st.session_state.current_model_name = model_choice

    # Language selection
    available_languages = ["sotho", "sepedi", "setswana"]
    current_language_index = available_languages.index(st.session_state.get("current_language", "sotho")) if st.session_state.get("current_language", "sotho") in available_languages else 0

    language_choice = st.sidebar.selectbox(
        "Select Language",
        available_languages,
        index=current_language_index,
        key="language_selector_sidebar"
    )
    st.session_state.current_language = language_choice

    # API Key input
    st.sidebar.header("API Key")
    if model_choice == "Claude":
        st.session_state.api_keys["claude"] = st.sidebar.text_input("Claude API Key", type="password", value=st.session_state.api_keys.get("claude", ""))
    elif model_choice == "OpenAI":
        st.session_state.api_keys["openai"] = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.api_keys.get("openai", ""))
    elif model_choice == "Gemini":
        st.session_state.api_keys["gemini"] = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.api_keys.get("gemini", ""))
    elif model_choice == "Llama 4":
        st.session_state.api_keys["groq"] = st.sidebar.text_input("Groq API Key (for Llama 4)", type="password", value=st.session_state.api_keys.get("groq", ""))
        st.sidebar.caption("Llama 4 model runs via Groq. Get API key from [console.groq.com](https://console.groq.com/)")

    return model_choice, language_choice


def initialize_llm(model_choice, api_keys):
    """
    Initializes the selected LLM based on UI choice and returns the instance.
    Updates st.session_state.current_model.
    """
    model_to_instantiate = None
    api_key = None
    model_identifier = None
    error_message = None

    try:
        if model_choice == "Claude":
            api_key = api_keys.get("claude")
            if not api_key:
                error_message = "Claude API key is missing."
            elif ClaudeLLM:
                model_to_instantiate = ClaudeLLM

        elif model_choice == "OpenAI":
            api_key = api_keys.get("openai")
            if not api_key:
                error_message = "OpenAI API key is missing."
            elif OpenAILLM:
                model_to_instantiate = OpenAILLM

        elif model_choice == "Gemini":
            api_key = api_keys.get("gemini")
            if not api_key:
                error_message = "Gemini API key is missing."
            elif GeminiLLM:
                model_to_instantiate = GeminiLLM

        elif model_choice == "Llama 4":
            api_key = api_keys.get("groq")
            if not api_key:
                error_message = "Groq API key (for Llama 4) is missing."
            elif LlamaLLM:
                model_to_instantiate = LlamaLLM
                model_identifier = "meta-llama/llama-4-scout-17b-16e-instruct"

        else:
            error_message = f"Model '{model_choice}' not supported yet."

        if error_message:
            st.sidebar.warning(error_message)
            return None

        if model_to_instantiate is None:
            _llm_import_warning()
            st.sidebar.error(f"{model_choice} is not available. Check if dependencies were properly imported.")
            return None

        # Check for reinit
        current = st.session_state.get('current_model')
        needs_reinit = (
            current is None or
            not isinstance(current, model_to_instantiate) or
            (hasattr(current, 'api_key') and current.api_key != api_key) or
            (model_identifier and hasattr(current, 'model') and current.model != model_identifier)
        )

        if needs_reinit:
            if model_identifier:
                instance = model_to_instantiate(api_key=api_key, model=model_identifier)
            else:
                instance = model_to_instantiate(api_key=api_key)

            if hasattr(instance, 'setup_client') and callable(instance.setup_client):
                instance.setup_client()

            st.session_state.current_model = instance
            st.sidebar.success(f"{model_choice} model ready.")
            if model_choice == "Llama 4":
                st.sidebar.caption(f"Requesting: {model_identifier}")
                st.sidebar.caption("Note: Call will fail if this model isn't available on Groq.")

        return st.session_state.current_model

    except Exception as e:
        st.sidebar.error(f"Error initializing {model_choice}: {e}")
        return None


def export_dataframe(df, fmt):
    """Export dataframe to specified format and create a download link"""
    # Placeholder implementation
    pass
