# pages/04_Sentiment_Bearings.py
import streamlit as st
import pandas as pd
import os
import traceback # For detailed error logging

# --- Page Configuration ---
st.set_page_config(page_title="Sentiment Bearings", layout="wide")

# Import utilities for sidebar, LLM init, export, state check
try:
    # Assume these exist and work as before
    from utils.shared_ui import ensure_session_state, display_sidebar, initialize_llm, export_dataframe
except ImportError as e:
    st.error(f"Failed to import utilities from 'utils.shared_ui': {e}. Please ensure it exists.")
    st.stop()

# Import the NEW classifier class
try:
    from analysis.sentiment_classifications import SentimentBearingClassifier # Import the new class
    from analysis.sentiment_classifications import PYMUPDF_INSTALLED # Check if PDF processing is available
except ImportError as e:
    st.error(f"Failed to import SentimentBearingClassifier from 'analysis.sentiment_bearing': {e}")
    st.info("Please ensure 'analysis/sentiment_bearing.py' exists and contains the class.")
    st.stop()

st.title("Sentiment Bearing Classification")
st.markdown("Classify text sentiment as **Positive**, **Negative**, or **Neutral**.")

# --- Initialize Session State and Sidebar ---
ensure_session_state()
model_choice, language_choice = display_sidebar()
llm_model = initialize_llm(model_choice, st.session_state.api_keys)

# Initialize session state variables specific to this page
if 'bearing_input_text' not in st.session_state:
    st.session_state.bearing_input_text = ""
if 'single_bearing_result' not in st.session_state:
    st.session_state.single_bearing_result = None
if 'bearing_results_df' not in st.session_state:
    st.session_state.bearing_results_df = pd.DataFrame()
# Removed category selection states
if 'bearing_technique' not in st.session_state:
    st.session_state.bearing_technique = "Zero-Shot"
# Removed include_basic state

# --- Prompt Technique Selection ---
technique_col1, technique_col2 = st.columns([1, 2])
with technique_col1:
    st.subheader("Prompt Technique")
    bearing_technique_radio = st.radio(
        "Select prompt technique:",
        ["Zero-Shot", "Few-Shot"],
        index=["Zero-Shot", "Few-Shot"].index(st.session_state.bearing_technique),
        key="bearing_technique_radio_key", # Use unique key
        on_change=lambda: st.session_state.update(
            bearing_technique=st.session_state.bearing_technique_radio_key
        )
    )
    # Update state immediately (radio's on_change might be delayed)
    st.session_state.bearing_technique = bearing_technique_radio
with technique_col2:
    st.info(
        "**Zero-Shot**: Ask the model directly without examples.\n\n"
        "**Few-Shot**: Provide examples to potentially improve accuracy."
    )

# --- Sentiment Bearing Logic & UI ---
if not llm_model:
    st.warning(f"LLM ({model_choice}) not initialized. Please provide the API key in the sidebar.")
else:
    # Instantiate the classifier
    try:
        sentiment_classifier = SentimentBearingClassifier(llm_model)
    except Exception as class_init_error:
         st.error(f"Failed to initialize SentimentBearingClassifier: {class_init_error}")
         st.stop()


    # --- Main Analysis Interface ---
    input_col, results_col = st.columns([1, 2])

    with input_col:
        st.subheader("Input Source")
        source_options = ["Enter Text Manually",
                          f"From {language_choice.capitalize()} Bootstrapped Lexicon",
                          "Upload File (.txt, .csv" + (", .pdf" if PYMUPDF_INSTALLED else "") + ")"] # Conditionally add PDF

        # Define callback for source change
        def on_source_change():
            st.session_state.single_bearing_result = None
            st.session_state.bearing_results_df = pd.DataFrame()
            # Clear text area if source changes from manual input
            if st.session_state.get("bearing_source_key") != "Enter Text Manually":
                 st.session_state.bearing_input_text = ""

        analysis_source = st.radio(
            "Select source:",
            source_options,
            key="bearing_source_key", # Use unique key
            on_change=on_source_change
        )

        # --- Input Area Logic ---
        if analysis_source == "Enter Text Manually":
            st.markdown("**Enter Text Manually**")
            st.session_state.bearing_input_text = st.text_area(
                "Text to Classify:", value=st.session_state.bearing_input_text,
                height=150, key="bearing_text_area" # Unique key
            )
            analyze_text_button = st.button("Classify Text", key="bearing_analyze_text", type="secondary", use_container_width=True)
            if analyze_text_button:
                if st.session_state.bearing_input_text.strip():
                    text_to_analyze = st.session_state.bearing_input_text.strip()
                    prompt_tech = st.session_state.bearing_technique # Get current technique
                    with st.spinner(f"Classifying text using {model_choice} ({prompt_tech})..."):
                        try:
                            classification_result = sentiment_classifier.classify_bearing(
                                text_to_analyze, language_choice, prompt_tech
                            )
                            st.session_state.single_bearing_result = classification_result
                            st.session_state.bearing_results_df = pd.DataFrame() # Clear batch results
                        except Exception as e:
                            st.error(f"Error during classification: {e}")
                            st.session_state.single_bearing_result = None
                else:
                    st.warning("Please enter text to classify.")
                    st.session_state.single_bearing_result = None

        elif analysis_source == f"From {language_choice.capitalize()} Bootstrapped Lexicon":
            st.markdown(f"**Classify from {language_choice.capitalize()} Bootstrapped Lexicon**")
            lexicon_filename = f"{language_choice}_bootstrapped.csv"
            # Check existence (simple check, backend does more robust check)
            lexicon_exists = os.path.exists(lexicon_filename) or os.path.exists(f"data/{lexicon_filename}")
            if lexicon_exists:
                limit_analysis = st.number_input("Limit analysis to first N words (0=all):", min_value=0, value=10, key="bearing_lex_limit")
                analyze_lexicon_button = st.button("Classify Bootstrapped Lexicon", key="bearing_analyze_lexicon", type="secondary", use_container_width=True)
                if analyze_lexicon_button:
                    limit = limit_analysis if limit_analysis > 0 else None
                    prompt_tech = st.session_state.bearing_technique
                    with st.spinner(f"Classifying lexicon using {prompt_tech}... (Limit: {'All' if limit is None else limit})"):
                        try:
                            results_df_batch = sentiment_classifier.classify_bootstrapped_lexicon(
                                language_choice, limit=limit, prompt_technique=prompt_tech
                            )
                            if results_df_batch is not None and not results_df_batch.empty and "error" not in results_df_batch.columns:
                                st.session_state.bearing_results_df = results_df_batch
                                st.session_state.single_bearing_result = None
                                st.success(f"Classified {len(results_df_batch)} terms.")
                            elif results_df_batch is not None and "error" in results_df_batch.columns:
                                 st.error(f"Lexicon classification failed: {results_df_batch['error'].iloc[0]}")
                                 st.session_state.bearing_results_df = pd.DataFrame()
                            else:
                                st.error("Classification failed or no terms analyzed.")
                                st.session_state.bearing_results_df = pd.DataFrame()
                        except Exception as e:
                            st.error(f"Error during lexicon classification: {e}")
                            st.session_state.bearing_results_df = pd.DataFrame()
            else:
                st.warning(f"Bootstrapped lexicon file ('{lexicon_filename}' or 'data/{lexicon_filename}') not found.")

        elif analysis_source.startswith("Upload File"):
            st.markdown("**Classify from Uploaded File**")
            allowed_types = ["txt", "csv"]
            if PYMUPDF_INSTALLED: allowed_types.append("pdf") # Only allow PDF if library is installed

            uploaded_file = st.file_uploader("Upload File:", type=allowed_types, key="bearing_file_uploader", label_visibility="collapsed")
            if uploaded_file:
                # Display file info
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
                st.json(file_details)

                analyze_file_button = st.button("Classify File Content", key="bearing_analyze_file", type="secondary", use_container_width=True)
                if analyze_file_button:
                    prompt_tech = st.session_state.bearing_technique
                    progress_bar = st.progress(0, text="Starting file classification...") # Add progress bar
                    with st.spinner(f"Classifying '{uploaded_file.name}' using {prompt_tech}... (This may take time for large files)"):
                        try:
                            # Ensure file pointer is at the beginning if reusing the object
                            uploaded_file.seek(0)
                            results_df_batch = sentiment_classifier.classify_uploaded_file(
                                uploaded_file, language_choice, prompt_tech
                            )
                            # Update progress bar upon completion
                            progress_bar.progress(100, text="Classification Complete.")

                            if results_df_batch is not None and not results_df_batch.empty and "error" not in results_df_batch.columns:
                                st.session_state.bearing_results_df = results_df_batch
                                st.session_state.single_bearing_result = None
                                st.success(f"Classified {len(results_df_batch)} items/chunks.")
                            elif results_df_batch is not None and "error" in results_df_batch.columns:
                                st.error(f"File classification failed: {results_df_batch['error'].iloc[0]}")
                                st.session_state.bearing_results_df = pd.DataFrame()
                            else:
                                st.error("Classification failed or no text extracted/processed.")
                                st.session_state.bearing_results_df = pd.DataFrame()
                        except Exception as e:
                            progress_bar.progress(100, text="Classification Failed.")
                            st.error(f"Error during file classification: {e}")
                            st.error(traceback.format_exc()) # Show detailed traceback in UI for debugging
                            st.session_state.bearing_results_df = pd.DataFrame()

    # --- Results Display Area ---
    with results_col:
        st.subheader("Classification Results")
        technique_badge = f"🔧 {st.session_state.bearing_technique}"
        st.write(f"{technique_badge} Classification")
        st.markdown("---")

        # Display Single Result
        if analysis_source == "Enter Text Manually":
            result = st.session_state.single_bearing_result
            if result:
                st.markdown("**Input Text:**"); st.text(st.session_state.bearing_input_text)
                st.markdown("**Sentiment Bearing:**")
                bearing = result.get('bearing', 'N/A').capitalize()
                is_valid = result.get('is_valid', False)

                if bearing == 'Positive': st.success(f"✅ {bearing}")
                elif bearing == 'Negative': st.error(f"❌ {bearing}")
                elif bearing == 'Neutral': st.info(f"⚪ {bearing}")
                elif bearing == 'Error': st.error(f"⚠️ {bearing}")
                else: st.write(bearing)

                explanation = result.get('explanation', "No explanation provided.")
                with st.expander("**Explanation:**", expanded=True): st.write(explanation)
                if not is_valid: st.warning("Classification may be invalid or incomplete.")

            # Message if button was clicked but no result (e.g., empty input)
            elif st.session_state.get("bearing_analyze_text"):
                 if not st.session_state.bearing_input_text.strip(): st.warning("Input text was empty.")
                 else: st.info("Classification finished, but no result was generated.")


        # Display Batch Results
        else:
            results_df = st.session_state.bearing_results_df
            if not results_df.empty and "error" not in results_df.columns:
                st.success(f"Displaying {len(results_df)} classification results.")

                # --- Results Summary (Optional) ---
                try:
                    if 'bearing' in results_df.columns:
                        bearing_counts = results_df['bearing'].value_counts()
                        with st.expander("Bearing Distribution Summary"):
                             # Use different colors for bearings in chart if possible
                             st.bar_chart(bearing_counts)
                except Exception as e: st.warning(f"Could not generate summary chart: {e}")
                st.markdown("---")

                # --- Results Table ---
                st.markdown("**Results Table:**")
                # Determine text column dynamically
                potential_text_cols = ['text', 'term', 'word', 'phrase']
                text_col = next((col for col in potential_text_cols if col in results_df.columns), results_df.columns[0] if results_df.columns.any() else None)

                display_cols = []
                if text_col: display_cols.append(text_col)
                if 'page' in results_df.columns: display_cols.append('page') # Add page for PDFs
                if 'bearing' in results_df.columns: display_cols.append('bearing')
                if 'is_valid' in results_df.columns: display_cols.append('is_valid')

                if display_cols:
                     st.dataframe(results_df[display_cols], height=300, use_container_width=True)
                else:
                     st.warning("No relevant columns found in results to display table.")

                # --- Explanation Viewer ---
                if 'explanation' in results_df.columns and text_col:
                     st.markdown("---"); st.write("**View Explanation by Item:**")
                     try:
                          # Create unique display options if text is duplicated
                          results_df['display_option'] = results_df[text_col].str[:80] + "..." + " (Index " + results_df.index.astype(str) + ")"
                          options_list = ["--- Select an item ---"] + results_df['display_option'].tolist()
                          selected_display_option = st.selectbox("Select item:", options=options_list, key="explanation_selector_bearing")

                          if selected_display_option and selected_display_option != options_list[0]:
                              # Find the original row based on the unique display option
                              selected_row_index = results_df[results_df['display_option'] == selected_display_option].index[0]
                              explanation = results_df.loc[selected_row_index, 'explanation']
                              item_text = results_df.loc[selected_row_index, text_col]
                              st.info(f"**Explanation for item (Index {selected_row_index}):**\n\n'{item_text[:200]}...'\n\n---\n{explanation}")
                     except Exception as e: st.warning(f"Could not display explanation selector: {e}")
                # else: st.caption("Explanation column not found.") # Uncomment if needed

                # --- Export ---
                st.markdown("---"); st.write("**Export Results:**")
                export_format = st.selectbox("Format", ["CSV", "Excel"], key="bearing_export_format") # Removed JSON for simplicity now
                export_button = st.button("Export Data", key="export_bearing_data")
                if export_button:
                    try:
                         # Export ALL columns, including explanations etc.
                         export_dataframe(results_df, "sentiment_bearing_results", fmt=export_format.lower())
                         # The export function itself should handle the download trigger
                         # st.success(f"Preparing {export_format} download...") # Button triggers download via data=
                    except Exception as e:
                         st.error(f"Export failed: {e}"); st.error(traceback.format_exc())

            # Placeholder if analysis ran but no results df
            elif st.session_state.get('bearing_analyze_lexicon') or \
                 st.session_state.get('bearing_analyze_file'):
                    if "error" in st.session_state.bearing_results_df.columns:
                         st.error(f"Analysis failed. Error: {st.session_state.bearing_results_df['error'].iloc[0]}")
                    else:
                         st.info("Analysis complete, but no results were generated (check file content or cleaning).")
