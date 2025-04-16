# pages/02_Lexicon_Generation.py
import streamlit as st
import pandas as pd
import os
import traceback

# Import utilities for sidebar, LLM init, export, state check
from utils.shared_ui import ensure_session_state, display_sidebar, initialize_llm, export_dataframe

# Import required classes (adjust paths if necessary)
try:
    from classes.generate_lexicons import GenerateLexicon
except ImportError:
    st.error("Failed to import GenerateLexicon class. Check file structure.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Lexicon Generation", layout="wide")
st.title("Lexicon Generation")
st.markdown("Generate sentiment lexicons using the selected LLM.")

# --- Initialize Session State and Sidebar ---
ensure_session_state()
model_choice, language_choice = display_sidebar()
llm_model = initialize_llm(model_choice, st.session_state.api_keys)

# --- Lexicon Generation Logic ---
def run_lexicon_generation(llm_model, language, categories):
    try:
        generator = GenerateLexicon(llm_model, language)
        combined_df = generator.generate_lexicon(categories=categories)
        return combined_df
    except Exception as e:
        st.error(f"Error during lexicon generation: {e}")
        print(f"Traceback for lexicon generation error:\n{traceback.format_exc()}")
        try:
            generator = GenerateLexicon(llm_model, language)
            return generator.load_existing_lexicon()
        except Exception:
            return pd.DataFrame(columns=["word", "meaning"])

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Configuration")
    categories_input_gen = st.text_input(
        "Focus Categories (comma-separated, optional)",
        placeholder="emotions, politics, economy",
        key="lex_categories"
    )
    categories_gen = [cat.strip() for cat in categories_input_gen.split(',')] if categories_input_gen else None
    
    generate_button_lex = st.button("Generate New Lexicon Entries", key="lex_generate_button", type="primary")

    if not llm_model:
        st.warning(f"LLM ({model_choice}) not initialized. Please provide the API key.")
    elif generate_button_lex:
        with st.spinner(f"Generating new entries for {language_choice} using {model_choice}..."):
            lexicon_df = run_lexicon_generation(
                llm_model=llm_model,
                language=language_choice,
                categories=categories_gen
            )
            if lexicon_df is not None:
                st.session_state.generated_lexicon = lexicon_df
                new_count = len(lexicon_df) - len(GenerateLexicon(llm_model, language_choice).load_existing_lexicon())
                st.success(f"Added {new_count} new entries to the lexicon!")

with col2:
    st.subheader("Current Lexicon")
    lexicon_file_path = f"data/{language_choice}.csv"
    try:
        if os.path.exists(lexicon_file_path):
            current_display_lexicon = pd.read_csv(lexicon_file_path)
            display_cols = [col for col in ['word', 'meaning'] if col in current_display_lexicon.columns]
            st.dataframe(current_display_lexicon[display_cols], height=400)

            st.markdown("---")
            st.write("Export Current Lexicon:")
            export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"], key="lex_export_format")
            if st.button("Export Lexicon File", key="lex_export_button"):
                data_to_export = current_display_lexicon[display_cols]
                data_lex = export_dataframe(data_to_export, export_format)
                if data_lex:
                    file_ext = {
                        "CSV": ".csv",
                        "Excel": ".xlsx", 
                        "JSON": ".json"
                    }[export_format]
                    
                    st.download_button(
                        label=f"Download {export_format}",
                        data=data_lex,
                        file_name=f"{language_choice}_lexicon{file_ext}",
                        mime="text/csv" if export_format == "CSV" else 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format == "Excel" else 
                             "application/json"
                    )
        else:
            st.info(f"No lexicon file found at {lexicon_file_path}. Generate entries to create it.")
    except Exception as load_err:
        st.error(f"Could not load lexicon file: {load_err}")
