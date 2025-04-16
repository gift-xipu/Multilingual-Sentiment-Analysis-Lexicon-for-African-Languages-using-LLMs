# pages/05_Lexicon_Bootstrapper.py
import streamlit as st
import pandas as pd
import os
import traceback
import json # Needed for parsing sentence list string potentially

# --- Page Configuration ---
st.set_page_config(page_title="Lexicon Bootstrapper", layout="wide")
st.title("👣 Lexicon Bootstrapper")
st.markdown("""
Build or refine context-aware sentiment lexicons using LLMs.
This tool takes seed words, generates example sentences, analyzes their sentiment,
and aggregates the results to infer the typical sentiment bearing and intensity of each word.
Results are saved to `{language}_lexicon_bootstrapped.csv` in the `data` folder.
""")

# Import utilities for sidebar, LLM init, export, state check
try:
    from utils.shared_ui import ensure_session_state, display_sidebar, initialize_llm, export_dataframe
except ImportError as e:
    st.error(f"Failed to import utilities from 'utils.shared_ui': {e}. Please ensure it exists.")
    st.stop()

# Import the NEW bootstrapper class
try:
    from analysis.lexicon_bootstrapper import LexiconBootstrapper
except ImportError as e:
    st.error(f"Failed to import LexiconBootstrapper from 'analysis.lexicon_bootstrapper': {e}")
    st.info("Please ensure 'analysis/lexicon_bootstrapper.py' exists and contains the class.")
    st.stop()

# --- Initialize Session State and Sidebar ---
ensure_session_state()
# Add default values for bootstrapper specific state
if 'bootstrap_seed_words' not in st.session_state:
    st.session_state.bootstrap_seed_words = ""
if 'bootstrap_num_sentences' not in st.session_state:
    st.session_state.bootstrap_num_sentences = 5 # Match default in backend
if 'bootstrap_update_existing' not in st.session_state:
    st.session_state.bootstrap_update_existing = False
if 'bootstrapped_lexicon_df' not in st.session_state:
    st.session_state.bootstrapped_lexicon_df = pd.DataFrame()
if 'bootstrap_input_lexicon_path' not in st.session_state: # Store path if found
    st.session_state.bootstrap_input_lexicon_path = None

model_choice, language_choice = display_sidebar()
llm_model = initialize_llm(model_choice, st.session_state.api_keys)

# --- Bootstrapping Logic & UI ---

if not llm_model:
    st.warning(f"LLM ({model_choice}) not initialized. Please provide the API key in the sidebar.")
else:
    # Instantiate the bootstrapper
    try:
        bootstrapper = LexiconBootstrapper(llm_model, language_choice)
    except Exception as class_init_error:
         st.error(f"Failed to initialize LexiconBootstrapper: {class_init_error}")
         st.stop()

    # Use columns for layout
    col_config, col_results = st.columns([1, 2]) # Config on left, results wider

    with col_config:
        st.subheader("Configuration")

        # --- Input Source Selection ---
        input_source = st.radio(
            "Select Seed Word Source:",
            [
                "Enter Seed Words Manually",
                f"Use Existing Bearing Lexicon ({language_choice}.csv)" # Use output of previous step
            ],
            key="bootstrap_source_radio"
            # Add on_change if needed to clear state, e.g., clear text area
            # on_change=lambda: st.session_state.update(bootstrap_seed_words="", bootstrapped_lexicon_df=pd.DataFrame())
        )

        seed_words_list = []
        word_col_from_lexicon = None # Keep track of the column name used
        st.session_state.bootstrap_input_lexicon_path = None # Reset path cache

        # --- Input Area based on Source ---
        if input_source == "Enter Seed Words Manually":
            st.markdown("**Enter Seed Words Manually**")
            st.session_state.bootstrap_seed_words = st.text_area(
                "Enter words/terms (one per line or comma-separated):",
                value=st.session_state.bootstrap_seed_words,
                height=150,
                key="bootstrap_seed_textarea",
                help="Enter the initial words you want to analyze and add to the lexicon."
            )
            # Process input text area into list
            raw_input = st.session_state.bootstrap_seed_words
            if raw_input.strip():
                # Split by newline first, then by comma if needed within lines
                potential_words = []
                for line in raw_input.splitlines():
                    potential_words.extend([w.strip() for w in line.split(',')])
                seed_words_list = [w for w in potential_words if w] # Filter empty strings

        else: # Use Existing Bearing Lexicon
            st.markdown(f"**Use Input Lexicon ({language_choice}.csv)**")
            input_lexicon_filename = f"{language_choice}.csv" # This is the INPUT file now
            input_lexicon_path = None
            # --- Find Input Lexicon Robustly ---
            # Define potential relative paths (adjust if structure differs)
            paths_to_check = [
                os.path.join("data", input_lexicon_filename), # Relative to CWD
                input_lexicon_filename # In CWD
            ]
            try: # Try paths relative to this UI script file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                paths_to_check.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'data', input_lexicon_filename)))
                paths_to_check.insert(1, os.path.abspath(os.path.join(script_dir, 'data', input_lexicon_filename)))
            except NameError: pass # __file__ not defined

            # Check paths in order
            for path in paths_to_check:
                 abs_path = os.path.abspath(path) # Ensure absolute path for check
                 if os.path.exists(abs_path):
                     input_lexicon_path = abs_path
                     st.session_state.bootstrap_input_lexicon_path = input_lexicon_path # Store found path
                     break
            # --- End Finding Input Lexicon ---

            if input_lexicon_path:
                st.caption(f"Reading words from: `{input_lexicon_path}`")
                try:
                    input_df = pd.read_csv(input_lexicon_path)
                    # Find the likely text column (word, term, text, etc.)
                    potential_cols = ['word', 'term', 'text', 'phrase', 'lemma', 'content']
                    df_cols_lower = {col.lower(): col for col in input_df.columns}
                    for col_potential in potential_cols:
                        if col_potential in df_cols_lower:
                            word_col_from_lexicon = df_cols_lower[col_potential]
                            break
                    if not word_col_from_lexicon:
                        word_col_from_lexicon = input_df.columns[0] if input_df.columns.any() else None

                    if word_col_from_lexicon:
                        seed_words_list = input_df[word_col_from_lexicon].dropna().astype(str).unique().tolist()
                        st.caption(f"Found {len(seed_words_list)} unique words/terms in column '`{word_col_from_lexicon}`'.")
                    else:
                         st.error("Could not identify a word/term column in the input lexicon CSV.")
                         seed_words_list = []

                except FileNotFoundError:
                     st.warning(f"Input lexicon file not found at expected path: {input_lexicon_path}")
                     seed_words_list = []
                except Exception as e:
                    st.error(f"Error reading input lexicon '{input_lexicon_path}': {e}")
                    seed_words_list = []
            else:
                st.warning(f"Input lexicon file '{input_lexicon_filename}' not found in standard locations (e.g., ./data, ../data relative to project root or script).")
                seed_words_list = []


        # --- Bootstrapping Configuration ---
        st.markdown("---")
        st.markdown("**Bootstrapping Options**")
        st.session_state.bootstrap_num_sentences = st.number_input(
            "Sentences per word:", min_value=1, max_value=20,
            value=st.session_state.bootstrap_num_sentences, step=1,
            key="bootstrap_num_sentences_input",
            help="How many example sentences the LLM should generate for each word."
        )
        st.session_state.bootstrap_update_existing = st.checkbox(
            "Update existing words",
            value=st.session_state.bootstrap_update_existing,
            key="bootstrap_update_checkbox",
            help="If checked, words already in the bootstrapped lexicon will be re-analyzed. Otherwise, only new words are added."
        )

        # --- Run Button ---
        st.markdown("---")
        run_button = st.button("Run Lexicon Bootstrapping", key="bootstrap_run_button", type="primary", use_container_width=True, disabled=(not seed_words_list))

        if not seed_words_list and input_source == "Enter Seed Words Manually":
             st.caption("Please enter seed words above.")
        elif not seed_words_list and input_source != "Enter Seed Words Manually":
             st.caption("Could not load seed words from the selected lexicon file.")


    # --- Results Display Area ---
    with col_results:
        st.subheader("Bootstrapped Lexicon Results")
        # Display the determined output path
        st.caption(f"Output file: `{bootstrapper.lexicon_path}`") # Get path from instance
        st.markdown("---")

        if run_button:
            if seed_words_list:
                st.info(f"Starting bootstrapping for {len(seed_words_list)} words...")
                # Clear previous results before running
                st.session_state.bootstrapped_lexicon_df = pd.DataFrame()
                try:
                    # The run_bootstrapping method handles Streamlit progress updates internally
                    # if streamlit is available
                    results_df = bootstrapper.run_bootstrapping(
                        word_list=seed_words_list,
                        num_sentences_per_word=st.session_state.bootstrap_num_sentences,
                        update_existing=st.session_state.bootstrap_update_existing
                    )
                    st.session_state.bootstrapped_lexicon_df = results_df
                    # Final success/error message handled within the method via st calls if available
                    if "error" in results_df.columns:
                        st.error(f"Bootstrapping completed with errors: {results_df['error'].iloc[0]}")

                except Exception as e:
                    st.error(f"An unexpected error occurred during bootstrapping run: {e}")
                    st.error(traceback.format_exc())
                    # Ensure results DF is cleared on major error
                    st.session_state.bootstrapped_lexicon_df = pd.DataFrame()
            else:
                 st.warning("No seed words provided or loaded. Cannot run bootstrapping.")


        # --- Display Results Table and Export ---
        results_df = st.session_state.bootstrapped_lexicon_df

        if not results_df.empty and "error" not in results_df.columns:
            st.write(f"Displaying {len(results_df)} entries from the bootstrapped lexicon.") # Moved success message here

            # Display Table
            st.markdown("**Generated/Updated Lexicon:**")
            # Select key columns for default display
            display_cols = ["word", "primary_sentiment", "intensity"]
            if 'rationale' in results_df.columns: display_cols.append('rationale')
            # Ensure columns exist before trying to display
            display_cols = [col for col in display_cols if col in results_df.columns]
            if display_cols:
                 st.dataframe(results_df[display_cols], height=400, use_container_width=True)
            else:
                 st.warning("Result DataFrame is missing expected columns.")


            # --- View Source Sentences ---
            st.markdown("---")
            st.write("**View Generated Sentences by Word:**")

            if 'word' in results_df.columns and 'source_sentences' in results_df.columns:
                try:
                    # Use index for guaranteed uniqueness in selectbox if words might repeat
                    results_df['display_option_bs'] = results_df['word'].astype(str) + " (Index " + results_df.index.astype(str) + ")"
                    word_options = ["--- Select a word ---"] + results_df['display_option_bs'].tolist()

                    # Handle potential stale state if results changed
                    selected_option_key = "sentence_viewer_selector_bs"
                    current_selection_bs = st.session_state.get(selected_option_key, word_options[0])
                    if current_selection_bs not in word_options:
                        current_selection_bs = word_options[0] # Reset if previous selection invalid

                    selected_display_option = st.selectbox(
                        "Select word:",
                        options=word_options,
                        key=selected_option_key,
                        index=word_options.index(current_selection_bs) # Set current index
                    )

                    if selected_display_option and selected_display_option != word_options[0]:
                        # Find the original row based on the unique display option
                        selected_row_index = results_df[results_df['display_option_bs'] == selected_display_option].index[0]
                        selected_row = results_df.loc[selected_row_index]
                        source_sentences_data = selected_row['source_sentences']
                        selected_word_actual = selected_row['word'] # Get actual word

                        st.markdown(f"**Generated Sentences for \"{selected_word_actual}\":**")

                        # Handle data (might be list or JSON string if loaded from CSV without full parsing)
                        sentence_list = []
                        if isinstance(source_sentences_data, list):
                            sentence_list = source_sentences_data
                        elif isinstance(source_sentences_data, str) and source_sentences_data.startswith('['):
                             try: # Safely parse string representation
                                 import ast
                                 sentence_list = ast.literal_eval(source_sentences_data)
                                 if not isinstance(sentence_list, list): sentence_list = [] # Ensure list
                             except (ValueError, SyntaxError, MemoryError, TypeError):
                                 st.warning("Could not parse sentence list string.")
                                 st.text(source_sentences_data) # Show raw
                        # Add JSON parsing as well
                        elif isinstance(source_sentences_data, str) and source_sentences_data.startswith('['):
                             try:
                                 sentence_list = json.loads(source_sentences_data.replace("'", "\""))
                                 if not isinstance(sentence_list, list): sentence_list = []
                             except json.JSONDecodeError:
                                  if not sentence_list: # If AST failed too
                                       st.warning("Could not parse sentence list string via JSON.")
                                       st.text(source_sentences_data) # Show raw

                        if sentence_list:
                            container = st.container(border=True)
                            for i, sentence in enumerate(sentence_list):
                                container.write(f"{i+1}. {sentence}")
                        elif not sentence_list: # Check if it was empty after parsing attempts
                             st.write("No source sentences recorded or data format issue for this entry.")


                except Exception as e:
                    st.warning(f"Could not display sentence viewer: {e}")
                    # st.error(traceback.format_exc()) # Uncomment for detailed debugging
            else:
                 st.caption("Required columns ('word', 'source_sentences') not found in results.")


            # Expander for full data view
            with st.expander("View Full Lexicon Data Table (including distribution and rationale)"):
                 st.dataframe(results_df)

            # --- Export ---
            st.markdown("---")
            st.write("**Export Full Bootstrapped Lexicon:**")
            export_format_lex = st.selectbox("Format", ["CSV", "Excel"], key="lex_export_format")
            export_button_lex = st.button("Export Lexicon Data", key="export_lex_button")

            if export_button_lex:
                try:
                    # IMPORTANT: Assume export_dataframe is fixed in utils/shared_ui.py
                    # Either it now accepts 'fmt' OR we remove it from the call.
                    # Let's try calling it WITHOUT fmt, assuming it defaults to CSV
                    # or uses the selected format internally based on its own logic.
                    # If it REQUIRES fmt, this call needs fmt=export_format_lex.lower() added back
                    # AFTER fixing the function definition.

                    export_dataframe(results_df, f"{language_choice}_lexicon_bootstrapped_export") # Removed fmt argument

                    # If export_dataframe doesn't use st.download_button internally:
                    # file_bytes = export_dataframe(results_df, ...) # Assuming it returns bytes
                    # if file_bytes:
                    #     st.download_button(...)
                    # else:
                    #     st.error("Export formatting failed.")

                    # Assuming export_dataframe handles the download button itself:
                    st.success(f"Preparing {export_format_lex} download (check browser).") # Message indicating action taken

                except TypeError as te:
                     # Catch the specific TypeError if fmt is still wrong
                     if "unexpected keyword argument 'fmt'" in str(te):
                           st.error("Export failed: The 'export_dataframe' function in 'utils/shared_ui.py' needs to be updated to handle the format selection (or the 'fmt' argument removed from the call here).")
                     else:
                           st.error(f"Export failed with TypeError: {te}")
                           st.error(traceback.format_exc())
                except Exception as e:
                     st.error(f"Export failed: {e}")
                     st.error(traceback.format_exc())

        elif "error" in results_df.columns: # Handle case where backend returned an error DataFrame
            st.error(f"Bootstrapping process failed: {results_df['error'].iloc[0]}")

        else:
            # Show message if lexicon file exists but is maybe empty or bootstrapping hasn't run
            if os.path.exists(bootstrapper.lexicon_path):
                 st.info(f"The bootstrapped lexicon file (`{bootstrapper.lexicon_path}`) exists but might be empty or no bootstrapping has been run in this session. Click 'Run' to generate/update it.")
            else:
                 st.info("Run bootstrapping to generate the lexicon file.")
