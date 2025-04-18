# utils/shared_ui.py
import streamlit as st
import pandas as pd
import io
import os
import traceback # Import traceback for detailed error logging

# --- Try to import LLMs safely ---
ClaudeLLM, OpenAILLM, GeminiLLM, LlamaLLM = None, None, None, None
_llm_import_error = None  # Save error to warn later

try:
    # Adjust paths if your structure is different
    from models.access.claude import ClaudeLLM
    from models.access.openai import OpenAILLM
    from models.access.gemini import GeminiLLM
    from models.access.llama import LlamaLLM
    print("Successfully imported LLM classes.") # Debug print
except ImportError as e:
    _llm_import_error = e
    print(f"Error importing LLM classes: {e}") # Debug print
    print(f"Full import error traceback:\n{traceback.format_exc()}")


def _llm_import_warning():
    """Show warning if LLM classes failed to import"""
    if _llm_import_error:
        st.warning(f"Could not import all LLM classes ({_llm_import_error}). Check file paths, class names, and dependencies (`requirements.txt`).")


def ensure_session_state():
    """Initializes required session state variables if they don't exist."""
    defaults = {
        "api_keys": {"claude": "", "openai": "", "gemini": "", "groq": ""},
        "current_model_name": "Claude", # Default model selection
        "current_language": "sotho",   # Default language
        "current_model": None,         # Holds the initialized LLM instance
        "generated_lexicon": pd.DataFrame(columns=["word", "sentiment", "intensity"]),
        "sentiment_analysis_results": pd.DataFrame(),
        "sentiment_input_text": "",
        "single_analysis_result": None,
        "selected_explanation_item": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            print(f"Initialized session state variable '{key}'.") # Debug print


def display_sidebar():
    """Displays sidebar configuration and returns selected model name and language."""
    st.sidebar.header("Model Configuration")

    # Define available models and corresponding class/key names
    # Filter list based on successful imports
    AVAILABLE_MODELS_MAP = {}
    if ClaudeLLM:
        AVAILABLE_MODELS_MAP["Claude"] = {"class": ClaudeLLM, "key_name": "claude"}
    if OpenAILLM:
        AVAILABLE_MODELS_MAP["OpenAI"] = {"class": OpenAILLM, "key_name": "openai"}
    if GeminiLLM:
        AVAILABLE_MODELS_MAP["Gemini"] = {"class": GeminiLLM, "key_name": "gemini"}
    if LlamaLLM:
         # Assuming Llama runs via Groq here based on original code
        AVAILABLE_MODELS_MAP["Llama (Groq)"] = {"class": LlamaLLM, "key_name": "groq"}

    available_model_names = list(AVAILABLE_MODELS_MAP.keys())

    if not available_model_names:
        st.sidebar.error("No LLM models could be imported. Please check imports and installations.")
        _llm_import_warning() # Show specific import error if available
        return None, st.session_state.get("current_language", "sotho") # Return None for model choice

    # Ensure current selection is valid, otherwise default to first available
    current_model_name = st.session_state.get("current_model_name", available_model_names[0])
    if current_model_name not in available_model_names:
        current_model_name = available_model_names[0]
        st.session_state.current_model_name = current_model_name

    current_model_index = available_model_names.index(current_model_name)

    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        available_model_names,
        index=current_model_index,
        key="model_selector_sidebar"
    )
    # Update session state immediately if choice changes
    if st.session_state.current_model_name != model_choice:
        st.session_state.current_model_name = model_choice
        st.session_state.current_model = None # Force reinitialization on model change
        print(f"Model selection changed to: {model_choice}. Cleared current_model.") # Debug print


    # Language selection
    available_languages = ["sotho", "sepedi", "setswana"]
    current_language = st.session_state.get("current_language", "sotho")
    if current_language not in available_languages:
        current_language = "sotho"
        st.session_state.current_language = current_language

    current_language_index = available_languages.index(current_language)

    language_choice = st.sidebar.selectbox(
        "Select Language",
        available_languages,
        index=current_language_index,
        key="language_selector_sidebar"
    )
    st.session_state.current_language = language_choice

    # API Key input based on selected model
    st.sidebar.header("API Key")
    selected_model_info = AVAILABLE_MODELS_MAP.get(model_choice)

    if selected_model_info:
        key_name = selected_model_info["key_name"]
        label = f"{model_choice} API Key"
        if model_choice == "Llama (Groq)":
            label = "Groq API Key (for Llama)"
            st.sidebar.caption("Llama models run via Groq. Get API key from [console.groq.com](https://console.groq.com/)")

        # Use the specific key_name for storing/retrieving the key
        api_key_value = st.session_state.api_keys.get(key_name, "")
        entered_key = st.sidebar.text_input(
            label,
            type="password",
            value=api_key_value,
            key=f"api_key_input_{key_name}" # Unique key per input
        )

        # Update the specific key in the session state dict
        if st.session_state.api_keys.get(key_name, "") != entered_key:
            st.session_state.api_keys[key_name] = entered_key
            st.session_state.current_model = None # Force reinitialization if key changes
            print(f"API Key for {key_name} updated. Cleared current_model.") # Debug print
    else:
        st.sidebar.warning("Selected model is not available.")


    return model_choice, language_choice


def initialize_llm(model_choice: str | None, api_keys: dict):
    """
    Initializes the selected LLM based on UI choice and returns the instance.
    Updates st.session_state.current_model.
    Returns None if initialization fails or no model is selected.
    """
    if model_choice is None:
        print("initialize_llm called with no model choice.") # Debug print
        return None

    print(f"Attempting to initialize LLM: {model_choice}") # Debug print

    # Check if the model is already initialized and if the API key/model ID matches
    current_instance = st.session_state.get('current_model')
    model_info = AVAILABLE_MODELS_MAP.get(model_choice)

    if not model_info:
        st.sidebar.error(f"Configuration error: Model '{model_choice}' not found in available map.")
        return None

    model_to_instantiate = model_info["class"]
    key_name = model_info["key_name"]
    api_key = api_keys.get(key_name)

    # Specific model identifier (e.g., for Llama/Groq)
    model_identifier = None
    if model_choice == "Llama (Groq)":
        # You might want to allow selecting specific Llama models later
        model_identifier = LlamaLLM.DEFAULT_MODEL # Use default from class or make selectable

    # Check if API key is present
    if not api_key:
        error_message = f"{model_choice} API key is missing."
        st.sidebar.warning(error_message)
        print(error_message) # Debug print
        if current_instance and isinstance(current_instance, model_to_instantiate):
             st.session_state.current_model = None # Clear invalid model
             print("Cleared existing model due to missing API key.") # Debug print
        return None

    # --- Check if re-initialization is needed ---
    needs_reinit = True # Default to re-init unless checks pass
    if current_instance:
        is_correct_class = isinstance(current_instance, model_to_instantiate)
        has_matching_key = hasattr(current_instance, 'api_key') and current_instance.api_key == api_key
        has_matching_model_id = True # Assume true unless specific model ID matters
        if model_identifier and hasattr(current_instance, 'model'):
            has_matching_model_id = current_instance.model == model_identifier

        if is_correct_class and has_matching_key and has_matching_model_id:
            needs_reinit = False
            # Ensure client is set up if it somehow got unset
            if hasattr(current_instance, 'client') and current_instance.client is None and hasattr(current_instance, 'setup_client'):
                 print("Existing model instance found, but client not set up. Attempting setup...") # Debug print
                 try:
                     current_instance.setup_client()
                     print("Client setup successful for existing instance.") # Debug print
                 except Exception as e:
                     st.sidebar.error(f"Error setting up client for existing {model_choice} instance: {e}")
                     print(f"Full traceback during setup of existing instance:\n{traceback.format_exc()}")
                     st.session_state.current_model = None # Invalidate if setup fails
                     return None # Failed to setup existing client

            else:
                 print(f"Using existing initialized instance of {model_choice}.") # Debug print

    # --- Perform initialization if needed ---
    if needs_reinit:
        print(f"Needs reinitialization for {model_choice}. Creating new instance...") # Debug print
        st.session_state.current_model = None # Clear previous instance first
        instance = None
        try:
            # Prepare arguments for the constructor
            init_args = {"api_key": api_key}
            if model_identifier:
                 # Pass the specific model identifier if required (e.g., for Llama/Groq)
                 # Ensure the LLM class __init__ accepts 'model'
                 init_args["model"] = model_identifier

            # Instantiate the LLM class (calls __init__)
            instance = model_to_instantiate(**init_args)

            # Call setup_client AFTER successful instantiation
            if hasattr(instance, 'setup_client') and callable(instance.setup_client):
                print("Calling setup_client for new instance...") # Debug print
                instance.setup_client() # This is where the Anthropic client init happens

            # Store the successfully initialized instance
            st.session_state.current_model = instance
            print(f"Successfully initialized and set session state for {model_choice}.") # Debug print
            st.sidebar.success(f"{model_choice} model ready.")
            if model_choice == "Llama (Groq)" and model_identifier:
                st.sidebar.caption(f"Using model: {model_identifier}")
                # st.sidebar.caption("Note: Call will fail if this model isn't available on Groq.")

        except Exception as e:
            # Catch errors during instantiation or setup_client call
            error_msg = f"Error initializing {model_choice}: {e}"
            st.sidebar.error(error_msg)
            print(error_msg) # Log to console
            print(f"Full traceback during initialization:\n{traceback.format_exc()}") # Print full traceback
            st.session_state.current_model = None # Ensure model is None if init fails
            return None # Return None to indicate failure

    return st.session_state.current_model


def export_dataframe(df: pd.DataFrame, filename_base: str, fmt: str):
    """Export dataframe to specified format (CSV or Excel) and create a download link"""
    if df.empty:
        st.warning("Cannot export empty data.")
        return

    try:
        buffer = io.BytesIO()
        if fmt.lower() == 'csv':
            df.to_csv(buffer, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel compatibility
            mime = 'text/csv'
            filename = f"{filename_base}.csv"
        elif fmt.lower() == 'excel':
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            # buffer.seek(0) # Not needed for ExcelWriter context manager
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f"{filename_base}.xlsx"
        else:
            st.error(f"Unsupported export format: {fmt}")
            return

        buffer.seek(0) # Go to the beginning of the buffer
        st.download_button(
            label=f"Download as {fmt.upper()}",
            data=buffer,
            file_name=filename,
            mime=mime,
        )
    except Exception as e:
        st.error(f"Error exporting data to {fmt}: {e}")
        print(f"Full traceback during export:\n{traceback.format_exc()}")
