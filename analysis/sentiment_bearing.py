import pandas as pd
import json
import re
import os
import io
import traceback
from typing import List, Optional, Dict, Any
import time

# PDF Parsing - requires pip install pymupdf
try:
    import fitz  # PyMuPDF
    PYMUPDF_INSTALLED = True
except ImportError:
    PYMUPDF_INSTALLED = False
    print("Info: PyMuPDF not installed (pip install pymupdf). PDF processing support disabled.")

# Optional: Sentence Splitting - requires pip install nltk
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=True) # Download quietly
            print("NLTK 'punkt' downloaded.")
        except Exception as download_err:
            print(f"Warning: Error downloading NLTK 'punkt': {download_err}. Sentence tokenization might be basic.")
    NLTK_INSTALLED = True
except ImportError:
    NLTK_INSTALLED = False
    print("Info: NLTK not installed. Sentence splitting for large text chunks will use basic methods.")


class SentimentBearings:
    """
    Classifies text sentiment bearing (Positive, Negative, Neutral) with rating and explanation.
    Supports manual input, lexicon files ({language}.csv), and uploads (TXT, CSV, PDF).
    """
    def __init__(self, llm_model):
        """
        Initializes the classifier.

        Parameters:
        - llm_model: An LLM instance with a `.generate(prompt: str) -> str` method.
        """
        if not hasattr(llm_model, 'generate') or not callable(llm_model.generate):
            raise TypeError("llm_model must have a callable 'generate' method.")
        self.llm = llm_model
        self.default_bearings = ["positive", "negative", "neutral"]

        # Few-shot examples for bearing/rating/explanation
        self.bearing_examples = {
            "sesotho": [
                {"text": "Ke thabile haholo!", "sentiment": "positive", "rating": 5, "explanation": "Expresses strong happiness."},
                {"text": "Seporo se koetsoe bakeng sa tokiso.", "sentiment": "neutral", "rating": 1, "explanation": "A neutral factual statement about track closure."},
                {"text": "Ke ne ke sa rata filimi eo ho hang.", "sentiment": "negative", "rating": 2, "explanation": "Expresses dislike, indicating negative sentiment, but intensity is moderate."}
            ],
            "sepedi": [
                {"text": "Re leboga kudu ka mpho ye.", "sentiment": "positive", "rating": 4, "explanation": "Expresses strong gratitude for a gift."},
                {"text": "Go na le dikoloi tše dintši tseleng.", "sentiment": "neutral", "rating": 1, "explanation": "A neutral observation about traffic."},
                {"text": "Ke befetšwe ke go diega.", "sentiment": "negative", "rating": 4, "explanation": "Expresses anger due to delay, a strong negative emotion."}
            ],
            "setswana": [
                {"text": "A bo go le monate jang!", "sentiment": "positive", "rating": 5, "explanation": "Exclamation showing strong enjoyment or pleasure."},
                {"text": "Pegelo e tla romelwa ka Labotlhano.", "sentiment": "neutral", "rating": 1, "explanation": "A neutral statement about when a report will be sent."},
                {"text": "Ga ke itumelele ditlamorago.", "sentiment": "negative", "rating": 3, "explanation": "Expresses dissatisfaction with results, a clear negative sentiment."}
            ],
            "english": [
                {"text": "This is absolutely wonderful news!", "sentiment": "positive", "rating": 5, "explanation": "Strong positive reaction to news."},
                {"text": "The meeting is scheduled for 3 PM.", "sentiment": "neutral", "rating": 1, "explanation": "A neutral, factual statement about scheduling."},
                {"text": "I am quite disappointed with the outcome.", "sentiment": "negative", "rating": 3, "explanation": "Expresses disappointment, a clear negative sentiment of moderate intensity."}
            ]
        }

    def clean_text(self, text: str) -> Optional[str]:
        """Cleans text, returns cleaned string or None."""
        if not isinstance(text, str) or not text.strip(): return None
        text = text.strip()
        boilerplate_patterns = [
            r'(?i)\bpage \d+(\s+of\s+\d+)?\b', r'^\s*\d+\s*$',
            r'(?i)^chapter \d+', r'(?i)^figure \d+', r'(?i)^table \d+',
            r'(?i)confidential', r'internal use only',
            # Add patterns specific to your data here
             r'(?i)e tsheseditswe ke:', r'(?i)e tshwerwe ke:', r'(?i)balekane:'
        ]
        for pattern in boilerplate_patterns:
            if re.search(pattern, text): return None
        text = re.sub(r'\s+', ' ', text).strip()
        # Keep slightly shorter snippets now, as single words from lexicon are valid
        return text if len(text.split()) >= 1 else None


    def _create_bearing_prompt(self, text: str, language: str, prompt_technique: str) -> str:
        """Generates the LLM prompt for sentiment bearing, rating, and explanation."""
        technique = prompt_technique.lower()

        instruction = f"""Analyze the sentiment of the following text snippet written in {language}.
Determine the primary sentiment expressed (positive, negative, or neutral).
Assign an intensity rating from 1 (very weak/neutral) to 5 (very strong). Neutral sentiment MUST have a rating of 1.
Provide a brief explanation (1-2 sentences) justifying your sentiment and rating based on the text.

Respond ONLY with a valid JSON object containing these exact keys: "sentiment", "rating", "explanation".
Example JSON: {{"sentiment": "positive", "rating": 4, "explanation": "The text expresses strong approval."}}
"""

        examples_str = ""
        if technique == "few_shot":
            examples = self.bearing_examples.get(language.lower(), self.bearing_examples["english"])
            examples_str = "\nHere are some examples of how to analyze and format the output:\n"
            for ex in examples[:2]: # Limit examples
                if all(k in ex for k in ['text', 'sentiment', 'rating', 'explanation']):
                    # Format example analysis as JSON string for the prompt
                    example_json = json.dumps({
                        "sentiment": ex['sentiment'],
                        "rating": ex['rating'],
                        "explanation": ex['explanation']
                    })
                    examples_str += f'\nText: "{ex["text"]}"\nAnalysis JSON: {example_json}\n'
            examples_str += "\nNow analyze the new text below and provide ONLY the JSON output:\n"

        prompt = f"""{instruction}
{examples_str}
--- TEXT TO ANALYZE ---
"{text}"
--- END TEXT ---

ANALYSIS JSON:
"""
        return prompt

    def _parse_bearing_response(self, response_raw: str) -> Dict[str, Any]:
        """Parses the LLM JSON response for bearing, rating, explanation."""
        result = {
            'sentiment': 'neutral',
            'rating': 1,
            'explanation': 'Failed to parse response.',
            'is_valid': False,
            'error_message': 'No valid JSON found.'
        }
        if not isinstance(response_raw, str) or not response_raw.strip():
            result['error_message'] = 'Empty response from LLM.'
            return result

        try:
            # Attempt to find and parse JSON
            match = re.search(r'\{\s*".*?".*?\}', response_raw, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    sentiment = str(data.get('sentiment', 'neutral')).lower().strip()
                    rating_raw = data.get('rating', 1 if sentiment == 'neutral' else 3) # Sensible defaults
                    explanation = str(data.get('explanation', 'No explanation provided.')).strip()

                    # Validate Sentiment
                    if sentiment not in self.default_bearings:
                         print(f"Warning: LLM returned invalid sentiment '{sentiment}'. Defaulting to neutral.")
                         sentiment = 'neutral' # Correct invalid sentiment

                    # Validate Rating
                    try: rating = int(round(float(rating_raw)))
                    except (ValueError, TypeError): rating = 1 if sentiment == 'neutral' else 3 # Default on conversion error
                    rating = max(1, min(5, rating)) # Clamp to 1-5

                    # Enforce neutral rating
                    if sentiment == 'neutral': rating = 1

                    result.update({
                        'sentiment': sentiment,
                        'rating': rating,
                        'explanation': explanation if explanation else "No explanation provided.",
                        'is_valid': True, # Parsed successfully
                        'error_message': None
                    })
                    return result

                except json.JSONDecodeError as json_err:
                    result['error_message'] = f"JSON Decode Error: {json_err}"
                    print(f"JSON Decode Error: {json_err} in '{json_str[:100]}...'")
                    # Fallback to regex if JSON parsing fails might be added here if needed
                except Exception as e: # Catch other potential errors during parsing
                     result['error_message'] = f"Parsing Error: {e}"
                     print(f"Error processing parsed JSON: {e}")

            else: # No JSON object found
                 print(f"Warning: Could not find JSON object in response: {response_raw[:200]}...")
                 # Simple Regex Fallback (less reliable)
                 response_lower = response_raw.lower()
                 sentiment = 'neutral'
                 if 'positive' in response_lower: sentiment = 'positive'
                 elif 'negative' in response_lower: sentiment = 'negative'
                 result['sentiment'] = sentiment
                 result['rating'] = 1 if sentiment == 'neutral' else 3 # Basic default rating
                 result['explanation'] = "Fallback: Could not parse structured explanation."
                 result['is_valid'] = False # Mark as invalid due to parsing failure

        except Exception as e:
            result['error_message'] = f"General Parsing Exception: {e}"
            print(f"Unexpected error during response parsing: {e}")
            traceback.print_exc()

        return result


    def analyze_sentiment(self, text: str, language: str, prompt_technique: str = "zero-shot") -> Dict[str, Any]:
        """
        Analyzes sentiment bearing, rating, and explanation for a single text string.
        Matches the method called by the Streamlit UI for manual input.
        """
        start_time = time.time()
        if not isinstance(text, str) or not text.strip():
            return {'sentiment': 'neutral', 'rating': 1, 'explanation': 'Input text is empty.', 'is_valid': False, 'error_message': 'Empty input'}

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {'sentiment': 'neutral', 'rating': 1, 'explanation': 'Text filtered out by cleaning.', 'is_valid': False, 'error_message': 'Cleaned text is empty'}

        prompt = self._create_bearing_prompt(cleaned_text, language, prompt_technique)

        try:
            print(f"Calling LLM for: '{cleaned_text[:60]}...' ({prompt_technique})")
            response_raw = self.llm.generate(prompt)
            llm_time = time.time()
            print(f"  LLM call took {llm_time - start_time:.2f}s")

            if not isinstance(response_raw, str): response_raw = str(response_raw)

            parsed_result = self._parse_bearing_response(response_raw)
            parse_time = time.time()
            print(f"  Parsing took {parse_time - llm_time:.2f}s. Valid: {parsed_result.get('is_valid')}")

            # Add original/cleaned text if needed by UI (optional)
            # parsed_result['original_text'] = text
            # parsed_result['analyzed_text'] = cleaned_text
            return parsed_result

        except Exception as e:
            print(f"Error during LLM call/processing for text '{cleaned_text[:60]}...': {e}")
            traceback.print_exc()
            return {
                'sentiment': 'error', 'rating': 1,
                'explanation': f"LLM call failed: {e}",
                'is_valid': False, 'error_message': str(e)
            }

    def _get_text_chunks(self, text: str, max_chunk_chars: int = 1500, min_chunk_length: int = 10) -> List[str]:
        """Splits large text into manageable chunks."""
        if not text: return []
        # Simple case: already short enough
        if len(text) <= max_chunk_chars:
            return [text] if len(text) >= min_chunk_length else []

        chunks = []
        # Try splitting by paragraphs first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is too long, try sentence splitting
            if len(para) > max_chunk_chars:
                sentences = []
                if NLTK_INSTALLED:
                    try: sentences = nltk.sent_tokenize(para)
                    except Exception as nltk_err: sentences = re.split(r'(?<=[.?!])\s+', para)
                else: sentences = re.split(r'(?<=[.?!])\s+', para)

                temp_chunk_sentences = []
                temp_chunk_len = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent: continue
                    if temp_chunk_len > 0 and temp_chunk_len + len(sent) + 1 > max_chunk_chars:
                        chunk_to_add = " ".join(temp_chunk_sentences)
                        if len(chunk_to_add) >= min_chunk_length: chunks.append(chunk_to_add)
                        temp_chunk_sentences = [sent]; temp_chunk_len = len(sent)
                    else:
                        temp_chunk_sentences.append(sent); temp_chunk_len += len(sent) + 1
                if temp_chunk_sentences:
                    chunk_to_add = " ".join(temp_chunk_sentences)
                    if len(chunk_to_add) >= min_chunk_length: chunks.append(chunk_to_add)

            # If paragraph fits, add it to the current chunk or start new one
            else:
                if len(para) < min_chunk_length: continue # Skip very short paragraphs
                if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if len(current_chunk) >= min_chunk_length: chunks.append(current_chunk)
                    current_chunk = para

        if len(current_chunk) >= min_chunk_length: chunks.append(current_chunk)

        # Final check for oversized chunks (force split)
        final_chunks = []
        for chunk in chunks:
             if len(chunk) > max_chunk_chars:
                  print(f"Warning: Chunk still >{max_chunk_chars} chars after splitting. Force splitting by length.")
                  final_chunks.extend([chunk[i:i+max_chunk_chars] for i in range(0, len(chunk), max_chunk_chars)])
             elif chunk:
                  final_chunks.append(chunk)

        return final_chunks


    def analyze_from_lexicon(self, language: str, limit: Optional[int] = None, prompt_technique: str = "zero-shot") -> pd.DataFrame:
        """
        Reads and analyzes terms from a lexicon file ({language}.csv).
        Matches the method called by the Streamlit UI.
        """
        results = []
        lexicon_filename = f"{language}.csv" # Use the filename specified in the UI
        paths_to_check = [
            lexicon_filename,
            os.path.join("data", lexicon_filename),
            os.path.join("..", "data", lexicon_filename) # Relative to script's parent dir
        ]
        try: # Add path relative to script itself
            script_dir = os.path.dirname(os.path.abspath(__file__))
            paths_to_check.insert(0, os.path.join(script_dir, '..', 'data', lexicon_filename))
        except NameError: pass # __file__ not defined

        actual_path_used = None
        for path in paths_to_check:
            if os.path.exists(path): actual_path_used = path; break

        if not actual_path_used:
            print(f"Error: Lexicon file not found for '{language}'. Checked: {paths_to_check}")
            return pd.DataFrame([{"error": f"Lexicon file '{lexicon_filename}' not found."}])

        print(f"Loading lexicon from: {actual_path_used}")
        try:
            df = pd.read_csv(actual_path_used)
            if df.empty: return pd.DataFrame([{"message": "Lexicon file is empty"}])

            # --- Identify Text Column ---
            text_col = None
            potential_cols = ['word', 'term', 'phrase', 'lemma', 'text', 'content']
            df_cols_lower = {col.lower(): col for col in df.columns}
            for col_potential in potential_cols:
               if col_potential in df_cols_lower: text_col = df_cols_lower[col_potential]; break
            if not text_col:
                if df.columns.any(): text_col = df.columns[0]
                else: return pd.DataFrame([{"error": "Lexicon CSV has no columns"}])
                print(f"Warning: Using first column '{text_col}' as text column for lexicon.")
            # --- End Text Column Identification ---

            df_to_process = df.head(limit) if limit and limit > 0 else df
            print(f"Analyzing {len(df_to_process)} terms from lexicon using '{prompt_technique}'...")

            # Add progress bar if in streamlit
            progress_bar = None
            try: progress_bar = st.progress(0.0, text=f"Analyzing lexicon (0/{len(df_to_process)})...")
            except Exception: pass # Not in streamlit

            for idx, row in df_to_process.iterrows():
                term = str(row[text_col]) if pd.notna(row[text_col]) else ""
                result_data = {text_col: term} # Start with original term

                if term.strip():
                    # Use the main analyze_sentiment method
                    classification_result = self.analyze_sentiment(term, language, prompt_technique)
                    result_data.update(classification_result)
                else:
                    result_data.update({'sentiment': 'neutral', 'rating': 1, 'explanation': 'Empty term in lexicon.', 'is_valid': False})

                results.append(result_data)
                if progress_bar:
                     progress_val = (idx + 1) / len(df_to_process)
                     progress_bar.progress(progress_val, text=f"Analyzing lexicon ({idx+1}/{len(df_to_process)})...")
                # time.sleep(0.05) # Optional delay

            print("Lexicon analysis finished.")
            if progress_bar: progress_bar.progress(1.0, text="Analysis complete.")
            return pd.DataFrame(results)

        except Exception as e:
             print(f"Error processing lexicon file {actual_path_used}: {e}")
             traceback.print_exc()
             return pd.DataFrame([{"error": f"Error processing lexicon: {e}"}])


    def analyze_uploaded_file(self, uploaded_file, language: str, prompt_technique: str = "zero-shot") -> pd.DataFrame:
        """
        Analyzes content from an uploaded file (PDF, TXT, CSV).
        Matches the method called by the Streamlit UI.
        """
        results = []
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()
        print(f"Processing uploaded file: {file_name} (Type: {ext})")

        try:
            # --- PDF Processing ---
            if ext == '.pdf':
                if not PYMUPDF_INSTALLED:
                    return pd.DataFrame([{"error": "PyMuPDF not installed, cannot process PDF."}])
                all_chunks_with_page = []
                pdf_doc = None # Initialize pdf_doc
                try:
                    pdf_doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                    print(f"Opened PDF with {len(pdf_doc)} pages.")
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc.load_page(page_num)
                        page_text = page.get_text("text", sort=True).strip()
                        cleaned_page_text = self.clean_text(page_text)
                        if cleaned_page_text:
                            chunks = self._get_text_chunks(cleaned_page_text)
                            for chunk in chunks:
                                all_chunks_with_page.append({"text": chunk, "page": page_num + 1})
                except Exception as pdf_err:
                     print(f"Error reading PDF content: {pdf_err}")
                     return pd.DataFrame([{"error": f"PDF Reading Error: {pdf_err}"}])
                finally:
                     if pdf_doc: pdf_doc.close() # Ensure closure

                if not all_chunks_with_page:
                    return pd.DataFrame([{"message": "No processable text extracted from PDF."}])

                print(f"Analyzing {len(all_chunks_with_page)} text chunks from PDF...")
                progress_bar = None
                try: progress_bar = st.progress(0.0, text=f"Analyzing PDF chunks (0/{len(all_chunks_with_page)})...")
                except Exception: pass

                for i, item in enumerate(all_chunks_with_page):
                    chunk_text = item["text"]
                    result_data = {"text": chunk_text, "page": item["page"]}
                    classification_result = self.analyze_sentiment(chunk_text, language, prompt_technique)
                    result_data.update(classification_result)
                    results.append(result_data)
                    if progress_bar:
                        progress_val = (i + 1) / len(all_chunks_with_page)
                        progress_bar.progress(progress_val, text=f"Analyzing PDF chunk {i+1}/{len(all_chunks_with_page)}...")
                    # time.sleep(0.05)

            # --- TXT Processing ---
            elif ext == '.txt':
                try:
                    content_bytes = uploaded_file.getvalue()
                    try: content = content_bytes.decode('utf-8')
                    except UnicodeDecodeError: content = content_bytes.decode('latin-1')
                except Exception as read_err:
                     return pd.DataFrame([{"error": f"TXT Reading Error: {read_err}"}])

                cleaned_content = self.clean_text(content) # Clean the whole content first
                if not cleaned_content:
                     return pd.DataFrame([{"message": "No processable text in TXT after cleaning."}])

                chunks = self._get_text_chunks(cleaned_content)
                if not chunks: return pd.DataFrame([{"message": "TXT resulted in no chunks after splitting."}])

                print(f"Analyzing {len(chunks)} text chunks from TXT...")
                progress_bar = None
                try: progress_bar = st.progress(0.0, text=f"Analyzing TXT chunks (0/{len(chunks)})...")
                except Exception: pass

                for i, chunk_text in enumerate(chunks):
                    result_data = {"text": chunk_text}
                    classification_result = self.analyze_sentiment(chunk_text, language, prompt_technique)
                    result_data.update(classification_result)
                    results.append(result_data)
                    if progress_bar:
                         progress_val = (i + 1) / len(chunks)
                         progress_bar.progress(progress_val, text=f"Analyzing TXT chunk {i+1}/{len(chunks)}...")
                    # time.sleep(0.05)

            # --- CSV Processing ---
            elif ext == '.csv':
                try:
                    bytes_data = uploaded_file.getvalue()
                    try: s = bytes_data.decode('utf-8')
                    except UnicodeDecodeError: s = bytes_data.decode('latin-1')
                    data = io.StringIO(s)
                    # Use low_memory=False for potentially mixed-type columns
                    df = pd.read_csv(data, low_memory=False)
                except Exception as read_err:
                     return pd.DataFrame([{"error": f"CSV Reading/Parsing Error: {read_err}"}])

                if df.empty: return pd.DataFrame([{"message": "CSV file is empty"}])

                # --- Identify Text Column ---
                text_col = None
                potential_cols = ['text', 'content', 'sentence', 'comment', 'review', 'phrase', 'word', 'term']
                df_cols_lower = {col.lower(): col for col in df.columns}
                for col_potential in potential_cols:
                   if col_potential in df_cols_lower: text_col = df_cols_lower[col_potential]; break
                if not text_col:
                    if df.columns.any(): text_col = df.columns[0]
                    else: return pd.DataFrame([{"error": "CSV has no columns"}])
                    print(f"Warning: Using first column '{text_col}' as text column for CSV.")
                # --- End Text Column Identification ---

                print(f"Analyzing {len(df)} rows from CSV (Column: '{text_col}')...")
                progress_bar = None
                try: progress_bar = st.progress(0.0, text=f"Analyzing CSV rows (0/{len(df)})...")
                except Exception: pass

                for idx, row in df.iterrows():
                     original_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                     result_data = {"text": original_text} # Keep original

                     if original_text.strip():
                          classification_result = self.analyze_sentiment(original_text, language, prompt_technique)
                          result_data.update(classification_result)
                     else:
                          result_data.update({'sentiment': 'neutral', 'rating': 1, 'explanation': 'Empty text in row.', 'is_valid': False})

                     results.append(result_data)
                     if progress_bar:
                         progress_val = (idx + 1) / len(df)
                         progress_bar.progress(progress_val, text=f"Analyzing CSV row {idx+1}/{len(df)}...")
                     # time.sleep(0.05)

            else:
                return pd.DataFrame([{"error": f"Unsupported file type: '{ext}'. Use PDF, TXT, or CSV."}])

            print("File analysis finished.")
            if progress_bar: progress_bar.progress(1.0, text="Analysis complete.")
            return pd.DataFrame(results)

        except Exception as e:
            print(f"An unexpected error occurred processing file '{file_name}': {e}")
            traceback.print_exc()
            return pd.DataFrame([{"error": f"Unexpected Error: {e}"}])
