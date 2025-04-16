# pages/03_Sentiment_Bearings.py
import streamlit as st
import pandas as pd
import os

# Import utilities for sidebar, LLM init, export, state check
from utils.shared_ui import ensure_session_state, display_sidebar, initialize_llm, export_dataframe

# Import required classes (adjust paths if necessary)
try:
    from analysis.sentiment_bearing import SentimentBearings
except ImportError:
    st.error("Failed to import SentimentBearings class. Check file structure.")
    st.stop() # Stop execution if class is critical

# --- Page Configuration ---
st.set_page_config(page_title="Sentiment Bearings", layout="wide") # Optional: Set page title
st.title("Sentiment Bearings Analysis")
st.markdown("Analyze text sentiment (Positive/Negative/Neutral), rating (1-5), and view explanations.")

# --- Initialize Session State and Sidebar ---
ensure_session_state()
model_choice, language_choice = display_sidebar()
llm_model = initialize_llm(model_choice, st.session_state.api_keys)

# --- Sentiment Bearings Logic & UI ---

if not llm_model:
    st.warning(f"LLM ({model_choice}) not initialized. Please provide the API key in the sidebar.")
else:
    # Instantiate the analyzer only if model is ready
    sentiment_analyzer = SentimentBearings(llm_model)

    # Use columns for layout
    col_input, col_results = st.columns([1, 2]) # Input on left, results wider

    with col_input:
        st.subheader("Input Source")
        # Use session state for radio button choice persistence might be good too
        analysis_source = st.radio(
            "Select:",
            ["Enter Text Manually", f"From {language_choice.capitalize()} Lexicon", "Upload File (.txt, .csv)"],
            key="sentiment_source",
            label_visibility="collapsed", # Hide label as subheader is present
            # Clear results when source changes
            on_change=lambda: st.session_state.update(
                single_analysis_result=None,
                sentiment_analysis_results=pd.DataFrame(),
                selected_explanation_item=None
            )
        )

        # --- Input Area based on Source ---
        if analysis_source == "Enter Text Manually":
            st.markdown("**Enter Text Manually**")
            st.session_state.sentiment_input_text = st.text_area(
                "Text to Analyze:", value=st.session_state.sentiment_input_text,
                height=100, key="sentiment_text_area"
            )
            analyze_text_button = st.button("Analyze Text", key="sentiment_analyze_text", type="primary", use_container_width=True)
            if analyze_text_button:
                if st.session_state.sentiment_input_text.strip():
                     text_to_analyze = st.session_state.sentiment_input_text.strip()
                     with st.spinner(f"Analyzing text using {model_choice}..."):
                         analysis_result = sentiment_analyzer.analyze_sentiment(text_to_analyze, language_choice)
                         st.session_state.single_analysis_result = analysis_result # Store result dict
                else:
                    st.warning("Please enter text to analyze.")
                    st.session_state.single_analysis_result = None

        elif analysis_source == f"From {language_choice.capitalize()} Lexicon":
             st.markdown(f"**Analyze from {language_choice.capitalize()} Lexicon**")
             lexicon_path = f"data/{language_choice}.csv" # Construct path directly
             if os.path.exists(lexicon_path):
                 limit_analysis = st.number_input("Limit analysis to first N words (0=all):", min_value=0, value=10, key="sentiment_lex_limit")
                 analyze_lexicon_button = st.button("Analyze Lexicon Words", key="sentiment_analyze_lexicon", type="primary", use_container_width=True)
                 if analyze_lexicon_button:
                     limit = limit_analysis if limit_analysis > 0 else None
                     with st.spinner(f"Analyzing words... (Limit: {'All' if limit is None else limit})"):
                         results_df_batch = sentiment_analyzer.analyze_from_lexicon(language_choice, limit=limit)
                     if results_df_batch is not None and not results_df_batch.empty:
                         st.session_state.sentiment_analysis_results = results_df_batch
                         st.session_state.selected_explanation_item = None
                         st.success(f"Analyzed {len(results_df_batch)} words.")
                     else:
                         st.error("Analysis failed or no words analyzed.")
                         st.session_state.sentiment_analysis_results = pd.DataFrame()
             else:
                 st.warning(f"Lexicon file not found: {lexicon_path}")

        elif analysis_source == "Upload File (.txt, .csv)":
            st.markdown("**Analyze from Uploaded File**")
            uploaded_file = st.file_uploader(
                "Upload File:", type=["txt", "csv"], key="sentiment_file_uploader", label_visibility="collapsed"
            )
            if uploaded_file:
                 analyze_file_button = st.button("Analyze File Content", key="sentiment_analyze_file", type="primary", use_container_width=True)
                 if analyze_file_button:
                     with st.spinner(f"Analyzing '{uploaded_file.name}'..."):
                         results_df_batch = sentiment_analyzer.analyze_uploaded_file(uploaded_file, language_choice)
                     if results_df_batch is not None and not results_df_batch.empty:
                         st.session_state.sentiment_analysis_results = results_df_batch
                         st.session_state.selected_explanation_item = None
                         st.success(f"Analyzed {len(results_df_batch)} items.")
                     else:
                         st.error("Analysis failed or no text extracted.")
                         st.session_state.sentiment_analysis_results = pd.DataFrame()

    # --- Results Display Area ---
    with col_results:
        st.subheader("Analysis Results")

        # --- Display Single Result ---
        if analysis_source == "Enter Text Manually":
            single_result = st.session_state.single_analysis_result
            if single_result:
                sentiment = single_result.get('sentiment', 'N/A').capitalize()
                rating = single_result.get('rating', 'N/A')
                explanation = single_result.get('explanation', "No explanation available.")

                res_col1, res_col2 = st.columns(2)
                with res_col1: st.metric(label="Sentiment", value=sentiment)
                with res_col2: st.metric(label="Rating (1-5)", value=str(rating))

                with st.expander("Show Explanation"):
                    st.info(explanation)
            else:
                st.info("Analysis results for manually entered text will appear here.")

        # --- Display Batch Results ---
        else:
            results_df = st.session_state.sentiment_analysis_results
            if not results_df.empty:
                text_col = 'word' if 'word' in results_df.columns else 'text'
                display_cols = [text_col, 'sentiment', 'rating']

                st.write("**Analysis Summary Table:**")
                st.dataframe(results_df[display_cols], height=300)
                st.markdown("---")

                # Explanation Section
                st.write("**View Explanation by Item:**")
                options = ["--- Select an item ---"] + results_df[text_col].unique().tolist()
                selected_item_key = "explanation_selector_batch" # Unique key

                # Check if previous selection is still valid, otherwise reset
                current_selection = st.session_state.get(selected_item_key, options[0])
                if current_selection not in options:
                     current_selection = options[0]

                selected_item = st.selectbox(
                    "Select item:", options=options, key=selected_item_key,
                    index=options.index(current_selection), # Set index based on current/reset value
                    label_visibility="collapsed"
                )
                st.session_state.selected_explanation_item = selected_item # Store selection

                if selected_item and selected_item != options[0]:
                    explanation_row = results_df[results_df[text_col] == selected_item].iloc[0]
                    explanation = explanation_row['explanation']
                    st.info(f"**Explanation for '{selected_item}':**\n\n{explanation}")

                st.markdown("---")
                # Export Full Results
                st.write("**Export Full Results:**")
                export_format_sb = st.selectbox("Format", ["CSV", "Excel", "JSON"], key="sb_export_format")
                if st.button("Export", key="sb_export_button"):
                    data_to_export = results_df # Export the full dataframe
                    data = export_dataframe(data_to_export, export_format_sb)
                    if data:
                        file_ext = ".csv" if export_format_sb == "CSV" else ".xlsx" if export_format_sb == "Excel" else ".json"
                        mime = "text/csv" if export_format_sb == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_format_sb == "Excel" else "application/json"
                        st.download_button(
                            label=f"Download {export_format_sb}", data=data,
                            file_name=f"{language_choice}_sentiment_analysis_{model_choice}_full{file_ext}",
                            mime=mime
                        )
            else:
                st.info("Batch analysis results (from Lexicon or File) will appear here.")
