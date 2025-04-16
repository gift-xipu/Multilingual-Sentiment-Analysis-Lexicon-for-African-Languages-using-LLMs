# analysis/lexicon_bootstrapper.py

import pandas as pd
import os
import re
import json
import traceback
import inspect
import time
from collections import Counter
import statistics
from typing import List, Optional, Dict, Any

# Using Streamlit for progress reporting if available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Define dummy st functions if not available
    class DummyStreamlit:
        def progress(self, *args, **kwargs): return self
        def text(self, *args, **kwargs): pass
        def empty(self): return self
        def success(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    st = DummyStreamlit()


# --- Configuration ---
DEFAULT_NUM_SENTENCES = 5       # How many sentences to generate per word
DEFAULT_ANALYSIS_RETRIES = 1    # How many times to retry sentence sentiment analysis on failure
DEFAULT_ANALYSIS_DELAY = 0.2    # Initial delay (seconds) before retry (uses exponential backoff)
# Define expected columns for the output bootstrapped lexicon
LEXICON_COLUMNS = ["word", "primary_sentiment", "intensity", "sentiment_distribution", "rationale", "source_sentences"]

class LexiconBootstrapper:
    """
    Implements Lexicon Bootstrapping using LLMs for sentence generation
    and sentiment analysis. Saves results to {language}_lexicon_bootstrapped.csv.
    """

    def __init__(self, llm_model, language):
        """
        Initializes the Bootstrapper.

        Args:
            llm_model: An initialized LLM interface object with a 'generate' method.
            language (str): The target language (e.g., 'sesotho', 'english').
        """
        if not hasattr(llm_model, 'generate') or not callable(llm_model.generate):
            raise TypeError("llm_model must have a callable 'generate' method.")
        self.llm_model = llm_model
        self.language = language.lower()

        # --- Determine Robust Path for saving the OUTPUT bootstrapped lexicon ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Assumes 'data' folder is one level up from 'analysis' directory
            self.base_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data'))
        except NameError: # Fallback if __file__ is not defined
            print("Warning: Could not determine script directory. Using 'data' relative to CWD.")
            self.base_data_path = "data"

        # Define the specific output filename
        self.lexicon_filename = f"{self.language}_lexicon_bootstrapped.csv" # Output filename
        self.lexicon_path = os.path.join(self.base_data_path, self.lexicon_filename)
        # --- End Robust Path ---

        print(f"Bootstrapper initialized for language '{self.language}'. Output lexicon path: '{self.lexicon_path}'")
        # Ensure the target directory exists
        try:
            os.makedirs(self.base_data_path, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create base data directory '{self.base_data_path}'. Saving might fail. Error: {e}")

    # --- Core Methods ---

    def _generate_sentences(self, word, num_sentences=DEFAULT_NUM_SENTENCES):
        """Generates synthetic sentences for a given word using the LLM."""
        print(f"    Generating {num_sentences} sentences for '{word}'...")
        prompt = f"""Generate exactly {num_sentences} diverse sentences in {self.language} that naturally use the word "{word}".
Show the word in varied contexts (positive, negative, neutral if possible).
Ensure sentences are grammatically correct and natural-sounding in {self.language}.
Output ONLY the sentences, each on a new line. No numbering, bullets, or extra text."""
        # Simple system prompt, adapt if needed for specific LLMs
        system_prompt = f"You are a helpful assistant fluent in {self.language}."

        try:
            response = self._call_llm(prompt, system_prompt=system_prompt)
            # Basic parsing and validation
            lines = [line.strip() for line in response.splitlines() if line.strip() and len(line) > 3]
            valid_sentences = [
                s for s in lines if word.lower() in s.lower() # Check if word is present
            ]
            if not valid_sentences and lines: # If word wasn't found but sentences were generated
                 print(f"    Warning: Sentences generated but word '{word}' not found in them. Trying to use generated lines anyway.")
                 valid_sentences = lines # Use the lines even if word check failed

            if len(valid_sentences) < num_sentences:
                print(f"    Warning: Generated only {len(valid_sentences)} valid-looking sentences for '{word}' (requested {num_sentences}).")
            if not valid_sentences:
                print(f"    Error: No valid sentences generated for '{word}'. LLM Response: '{response[:100]}...'")
                return []
            return valid_sentences[:num_sentences] # Return up to the requested number

        except Exception as e:
            print(f"    Error generating sentences for '{word}': {e}")
            traceback.print_exc()
            return []

    def _analyze_sentence_sentiment(self, sentence, retries=DEFAULT_ANALYSIS_RETRIES, delay=DEFAULT_ANALYSIS_DELAY):
        """Analyzes the sentiment of a single sentence using the LLM."""
        # This reuses the prompt/parsing logic similar to SentimentBearings class
        # Ensures consistency in how sentence sentiment is determined.
        print(f"        Analyzing sentence: '{sentence[:50]}...'")
        prompt = f"""Analyze the sentiment of the following {self.language} sentence:
\"\"\"
{sentence}
\"\"\"
Determine the sentiment polarity (positive, negative, or neutral) and provide an intensity rating from 1 (very weak/neutral) to 5 (very strong). Neutral sentiment MUST have an intensity rating of 1.

Provide your analysis ONLY in JSON format with keys "sentiment" (string: "positive", "negative", or "neutral") and "rating" (integer: 1-5). Example: {{"sentiment": "positive", "rating": 4}}
Return ONLY the valid JSON object. No explanations or other text."""
        system_prompt = "You are an AI expert in sentiment analysis. Provide your output strictly in the requested JSON format (keys: 'sentiment', 'rating')."

        last_error = None
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    print(f"        Retrying analysis (attempt {attempt + 1})...")
                    time.sleep(delay * (2 ** (attempt - 1))) # Exponential backoff

                response = self._call_llm(prompt, system_prompt=system_prompt)

                # Parse the JSON response
                match = re.search(r'\{\s*".*?".*?\}', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        data = json.loads(json_str)
                        sentiment = str(data.get('sentiment', '')).lower().strip()
                        rating_raw = data.get('rating')

                        if sentiment in ["positive", "negative", "neutral"] and isinstance(rating_raw, (int, float, str)):
                            try: rating_num = float(rating_raw) # Handle numbers as strings
                            except ValueError: rating_num = 1.0 if sentiment=='neutral' else 3.0

                            rating = max(1, min(5, round(rating_num)))
                            if sentiment == "neutral": rating = 1 # Enforce rating 1 for neutral

                            return {"sentiment": sentiment, "rating": rating} # Success
                        else:
                            last_error = f"Invalid data types or values in JSON: {data}"
                    except json.JSONDecodeError as json_err:
                        last_error = f"JSON Decode Error: {json_err} in '{json_str[:100]}...'"
                else:
                    last_error = f"Could not find valid JSON in response: '{response[:100]}...'"

            except Exception as e:
                last_error = f"LLM call failed during analysis: {e}"
                # Don't print traceback on every retry, only on final failure below
                if attempt == retries: traceback.print_exc()


        print(f"        Analysis failed after {retries + 1} attempts: {last_error}")
        return None # Indicate failure

    def _aggregate_results(self, word, sentence_analyses, source_sentences):
        """Aggregates sentence sentiments to infer word properties."""
        if not sentence_analyses: # Check if list is empty or contains only None
            return {
                "word": word, "primary_sentiment": "undetermined", "intensity": 0,
                "sentiment_distribution": {}, "rationale": "No valid sentence analyses.",
                "source_sentences": source_sentences # Still store the sentences generated
            }

        sentiments = [a['sentiment'] for a in sentence_analyses]
        ratings = [a['rating'] for a in sentence_analyses]

        counts = Counter(sentiments)
        # Ensure all potential sentiments are keys for consistent output structure
        dist = {
            s: counts.get(s, 0) / len(sentiments) for s in ["positive", "negative", "neutral"]
        }

        # Determine primary sentiment (majority, tie-break: pos > neg > neu)
        primary_sentiment = 'neutral' # Default
        max_count = 0
        if counts: max_count = max(counts.values())
        candidates = [s for s, c in counts.items() if c == max_count]

        if len(candidates) == 1: primary_sentiment = candidates[0]
        elif 'positive' in candidates: primary_sentiment = 'positive'
        elif 'negative' in candidates: primary_sentiment = 'negative'

        # Determine intensity
        intensity = 1 # Default to 1 (neutral or weak)
        average_rating = None
        if ratings:
            try:
                average_rating = statistics.mean(ratings)
                if primary_sentiment == 'neutral': intensity = 1
                else:
                    # Simple rounding, clamped, ensuring non-neutral isn't 1 unless avg rating is very low
                    calc_intensity = max(1, min(5, round(average_rating)))
                    intensity = max(2, calc_intensity) if calc_intensity == 1 and average_rating > 1.5 else calc_intensity
            except statistics.StatisticsError: intensity = 3 if primary_sentiment != 'neutral' else 1
        else: intensity = 3 if primary_sentiment != 'neutral' else 1

        # Rationale
        avg_rating_str = f"{average_rating:.2f}" if average_rating is not None else "N/A"
        rationale = (f"Aggregated from {len(sentiments)} valid sentence analyses. "
                     f"Counts: {dict(counts)}. Avg Rating: {avg_rating_str}. "
                     f"Inferred: Sent={primary_sentiment}, Int={intensity}.")

        return {
            "word": word,
            "primary_sentiment": primary_sentiment,
            "intensity": intensity,
            "sentiment_distribution": dist, # Store distribution as dict
            "rationale": rationale,
            "source_sentences": source_sentences # Store the sentences used
        }

    def _call_llm(self, prompt, system_prompt=None):
        """Helper function to call the LLM, handling system prompts."""
        try:
            # Basic check if generate might support system_prompt
            sig = inspect.signature(self.llm_model.generate)
            if system_prompt and 'system_prompt' in sig.parameters:
                response = self.llm_model.generate(prompt, system_prompt=system_prompt)
            # Check for Anthropic-style messages format
            elif system_prompt and hasattr(self.llm_model, 'messages') and callable(getattr(self.llm_model, 'invoke', None)): # Basic check for Langchain/Anthropic structure
                 response = self.llm_model.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                 if hasattr(response, 'content'): response = response.content # Extract content if it's an AIMessage or similar
            else: # Default to basic generate call
                 full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
                 response = self.llm_model.generate(full_prompt)

            return str(response).strip() if response is not None else ""
        except Exception as e:
            print(f"    LLM Error during call: {e}")
            raise # Re-raise the exception

    def _safe_literal_eval(self, val):
        """Safely evaluate string representations of lists/dicts from CSV."""
        if isinstance(val, (list, dict)): return val # Already correct type
        if not isinstance(val, str) or not val.strip():
             return {} if isinstance(val, dict) else [] if isinstance(val, list) else {} # Default based on expected type
        try:
            # Prioritize JSON loading
            return json.loads(val.replace("'", "\""))
        except (json.JSONDecodeError, TypeError):
             # Fallback using ast.literal_eval for basic list/dict structures if JSON fails
             try:
                  import ast
                  return ast.literal_eval(val)
             except (ValueError, SyntaxError, MemoryError, TypeError):
                  # If all parsing fails, return default type or original string? Decide behavior.
                  print(f"Warning: Could not parse value during load: {val[:100]}")
                  return {} if val.startswith('{') else [] if val.startswith('[') else val

    def _ensure_lexicon_schema(self, df):
        """Ensure DataFrame columns match LEXICON_COLUMNS and have correct types."""
        if df is None: return pd.DataFrame(columns=LEXICON_COLUMNS)

        # Add missing columns
        for col in LEXICON_COLUMNS:
            if col not in df.columns:
                default_val = {} if col == 'sentiment_distribution' else [] if col == 'source_sentences' else ''
                df[col] = default_val

        # Ensure correct order and select only defined columns
        df = df.reindex(columns=LEXICON_COLUMNS, fill_value='') # Use reindex for safety

        # Apply type conversions and constraints
        df['word'] = df['word'].astype(str).fillna('')
        df['primary_sentiment'] = df['primary_sentiment'].astype(str).fillna('undetermined')
        # Allow 0 intensity for undetermined/error cases
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=5)
        df['rationale'] = df['rationale'].astype(str).fillna('')
        # Ensure list/dict columns are evaluated correctly after potential merge/concat
        # Store as actual dicts/lists within the DataFrame during processing
        df['sentiment_distribution'] = df['sentiment_distribution'].apply(self._safe_literal_eval).apply(lambda x: x if isinstance(x, dict) else {})
        df['source_sentences'] = df['source_sentences'].apply(self._safe_literal_eval).apply(lambda x: x if isinstance(x, list) else [])

        return df

    # --- Public Orchestration Method ---

    def run_bootstrapping(self, word_list, num_sentences_per_word=DEFAULT_NUM_SENTENCES, update_existing=False):
        """
        Main method to run the bootstrapping process for a list of words.

        Args:
            word_list (list): List of strings (words to process).
            num_sentences_per_word (int): Number of sentences to generate per word.
            update_existing (bool): If True, re-process words already in the lexicon.

        Returns:
            pandas.DataFrame: The updated lexicon DataFrame.
        """
        print("\n--- Starting Lexicon Bootstrapping Run ---")
        # Load existing lexicon using the robustly determined path
        existing_df = pd.DataFrame(columns=LEXICON_COLUMNS) # Default empty
        if os.path.exists(self.lexicon_path):
            try:
                existing_df_raw = pd.read_csv(self.lexicon_path, encoding='utf-8')
                # Apply schema/parsing immediately after loading
                existing_df = self._ensure_lexicon_schema(existing_df_raw.copy()) # Process a copy
                print(f"Loaded {len(existing_df)} entries from existing lexicon: '{self.lexicon_path}'")
            except Exception as e:
                print(f"Error loading/parsing existing lexicon from '{self.lexicon_path}': {e}. Starting fresh.")
                existing_df = pd.DataFrame(columns=LEXICON_COLUMNS)
        else:
             print("No existing lexicon found. Starting fresh.")

        # Use lower case for comparison, keeping original case in 'word' column
        existing_words_lower = set(existing_df['word'].astype(str).str.lower()) if 'word' in existing_df.columns else set()

        # Prepare word list: unique, stripped strings
        words_to_process_raw = [str(w).strip() for w in word_list if isinstance(w, str) and str(w).strip()]
        unique_words = sorted(list(set(words_to_process_raw)), key=str.lower)

        # Filter words based on update_existing flag
        if update_existing:
            words_to_process = unique_words
            print(f"Processing {len(words_to_process)} unique words (update mode).")
        else:
            words_to_process = [w for w in unique_words if w.lower() not in existing_words_lower]
            print(f"Processing {len(words_to_process)} new unique words (excluding {len(existing_words_lower)} existing).")

        if not words_to_process:
            print("No words to process.")
            if STREAMLIT_AVAILABLE: st.info("No new words to process based on the provided list and existing lexicon.")
            return existing_df # Return unchanged lexicon

        # --- Process Words ---
        results = []
        total = len(words_to_process)
        # Use dummy progress bar if streamlit is not available
        progress_bar = st.progress(0.0) if STREAMLIT_AVAILABLE else None
        progress_text_widget = st.empty() if STREAMLIT_AVAILABLE else None
        if progress_text_widget: progress_text_widget.text(f"Starting bootstrapping for {total} words...")

        for i, word in enumerate(words_to_process):
            print(f"\nProcessing word {i+1}/{total}: '{word}'")
            if progress_bar and progress_text_widget:
                progress = (i / total)
                progress_bar.progress(progress)
                progress_text_widget.text(f"Processing '{word}' ({i+1}/{total})... Generating sentences.")

            # 1. Generate Sentences
            sentences = self._generate_sentences(word, num_sentences_per_word)
            if not sentences:
                results.append({ # Add entry indicating generation failure
                    "word": word, "primary_sentiment": "error", "intensity": 0,
                    "sentiment_distribution": {}, "rationale": "Failed to generate valid sentences.",
                    "source_sentences": []
                })
                continue

            # 2. Analyze Sentences
            if progress_text_widget: progress_text_widget.text(f"Processing '{word}' ({i+1}/{total})... Analyzing {len(sentences)} sentences.")
            analyses = [self._analyze_sentence_sentiment(s) for s in sentences]
            valid_analyses = [a for a in analyses if a is not None]

            # 3. Aggregate Results
            if progress_text_widget: progress_text_widget.text(f"Processing '{word}' ({i+1}/{total})... Aggregating results.")
            aggregated = self._aggregate_results(word, valid_analyses, sentences)
            results.append(aggregated)
            # time.sleep(0.1) # Optional delay

        # --- Combine and Save ---
        print("\n--- Combining results ---")
        if progress_text_widget: progress_text_widget.text("Combining results...")
        if not results: # No results were generated in the loop
             print("No results generated during processing.")
             final_df = existing_df # Return original DF
        else:
            new_results_df = pd.DataFrame(results)
            # Ensure schema before combining
            new_results_df = self._ensure_lexicon_schema(new_results_df.copy())

            if not new_results_df.empty:
                # Use lowercase word for matching/updating
                if 'word' in existing_df.columns:
                    existing_df['word_lower'] = existing_df['word'].astype(str).str.lower()
                else:
                    existing_df['word_lower'] = pd.Series(dtype=str)

                new_results_df['word_lower'] = new_results_df['word'].astype(str).str.lower()
                processed_words_lower = set(new_results_df['word_lower'])

                # Keep existing rows that were *not* processed in this run
                df_to_keep = existing_df[~existing_df['word_lower'].isin(processed_words_lower)].copy()

                # Concatenate the kept rows with the *new* results
                final_df = pd.concat([df_to_keep, new_results_df], ignore_index=True)

                # Final cleanup: Drop temp column, sort, drop duplicates
                final_df = final_df.drop(columns=['word_lower'], errors='ignore')
                # Sort before dropping duplicates to be deterministic if needed, keep latest results
                final_df = final_df.sort_values(by='word', key=lambda col: col.str.lower())
                final_df = final_df.drop_duplicates(subset=['word'], keep='last')
                # Reapply schema after all manipulations
                final_df = self._ensure_lexicon_schema(final_df.copy())

            else:
                print("No new results generated to combine.")
                final_df = existing_df.drop(columns=['word_lower'], errors='ignore') # Drop helper column if no new results

        # Convert dicts/lists to JSON strings *before* saving to CSV
        df_to_save = final_df.copy()
        for col in ['sentiment_distribution', 'source_sentences']:
            if col in df_to_save.columns:
                 # Handle potential errors during JSON conversion
                 df_to_save[col] = df_to_save[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        # Save the final DataFrame
        try:
            # The base_data_path directory existence is checked/created in __init__
            df_to_save.to_csv(self.lexicon_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for better Excel compatibility
            print(f"Successfully saved updated lexicon to '{self.lexicon_path}' ({len(final_df)} entries).")
            if progress_bar and progress_text_widget:
                progress_bar.progress(1.0)
                progress_text_widget.text(f"Bootstrapping complete! Lexicon saved ({len(final_df)} entries).")
                st.success(f"Bootstrapping complete! Lexicon updated/saved with {len(results)} results. Total entries: {len(final_df)}.")

        except Exception as e:
            print(f"Error saving final lexicon: {e}")
            traceback.print_exc()
            if progress_bar and progress_text_widget:
                 progress_bar.progress(1.0)
                 progress_text_widget.text("Bootstrapping complete, but save failed.")
                 st.error(f"Bootstrapping complete, but failed to save lexicon: {e}")

        print("--- Bootstrapping Run Finished ---")
        # Return the DataFrame with actual dicts/lists, not the JSON strings saved to CSV
        return final_df
