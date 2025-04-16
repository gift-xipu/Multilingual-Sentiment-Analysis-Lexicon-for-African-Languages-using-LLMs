import pandas as pd
import json
import re
import os
import io
import traceback
from typing import List, Optional, Dict, Any, Tuple
import time

# PDF Parsing - requires pip install pymupdf
try:
    import fitz  # PyMuPDF
    PYMUPDF_INSTALLED = True
except ImportError:
    PYMUPDF_INSTALLED = False
    print("Warning: PyMuPDF not installed (pip install pymupdf). PDF processing will be disabled.")

# Optional: Sentence Splitting - requires pip install nltk
try:
    import nltk
    # Attempt to download 'punkt' if not already available - requires internet on first run
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Attempting download...")
        nltk.download('punkt')
    NLTK_INSTALLED = True
except ImportError:
    NLTK_INSTALLED = False
    print("Warning: NLTK not installed. Sentence splitting for large text chunks will be basic.")

class SentimentBearingClassifier:
    """
    Classifies text sentiment bearing (Positive, Negative, Neutral)
    using an LLM, supporting manual input, lexicon files, and uploads
    (PDF, TXT, CSV).
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
        self.default_bearings = ["positive", "negative", "neutral"] # Explicitly define

        # Few-shot examples tailored for bearing classification
        self.bearing_examples = {
            "sesotho": [
                {"text": "Ke thabile haholo!", "bearing": "positive", "explanation": "Expresses clear happiness."},
                {"text": "Tsamaea butle tseleng ena.", "bearing": "neutral", "explanation": "A neutral instruction or advice."},
                {"text": "Ke maswabi haholo ka ditaba tseo.", "bearing": "negative", "explanation": "Expresses sadness or disappointment."}
            ],
            "sepedi": [
                {"text": "Ke leboga thušo ya gago kudu.", "bearing": "positive", "explanation": "Expresses gratitude, which is positive."},
                {"text": "Pula e a na ka maatla.", "bearing": "neutral", "explanation": "A neutral statement about the weather."},
                {"text": "Ga ke rate mokgwa woo.", "bearing": "negative", "explanation": "Expresses dislike or disapproval."}
            ],
            "setswana": [
                {"text": "Tiro e e dirilwe sentle tota!", "bearing": "positive", "explanation": "Positive praise for work done well."},
                {"text": "Buka e e kwadilwe ka 1990.", "bearing": "neutral", "explanation": "A neutral factual statement."},
                {"text": "Ga ke dumalane le wena.", "bearing": "negative", "explanation": "Expresses disagreement, indicating negativity."}
            ],
            "english": [
                {"text": "This is fantastic news!", "bearing": "positive", "explanation": "Clearly expresses positive excitement."},
                {"text": "The report is due on Friday.", "bearing": "neutral", "explanation": "A neutral statement of fact."},
                {"text": "I'm very unhappy with the service.", "bearing": "negative", "explanation": "Directly states unhappiness, which is negative."}
            ]
        }

    def clean_text(self, text: str) -> Optional[str]:
        """
        Cleans text by removing common artifacts, extra whitespace.
        Returns cleaned text or None if it should be skipped.
        """
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()
        # --- Add your specific cleaning rules here ---
        # Remove typical PDF headers/footers (customize patterns)
        boilerplate_patterns = [
            r'(?i)\bpage \d+(\s+of\s+\d+)?\b', r'^\s*\d+\s*$', # Page numbers
            r'(?i)^chapter \d+', r'(?i)^figure \d+', r'(?i)^table \d+',
            r'(?i)confidential', r'internal use only'
        ]
        for pattern in boilerplate_patterns:
             if re.search(pattern, text): return None # Skip if pattern matches
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Skip if too short after cleaning
        return text if len(text.split()) > 2 else None

    def _create_bearing_prompt(self, text: str, language: str, prompt_technique: str) -> str:
        """Generates the LLM prompt for sentiment bearing classification."""
        technique = prompt_technique.lower()
        bearings_str = ", ".join(self.default_bearings)

        instruction = f"""Analyze the sentiment bearing of the following text snippet written in {language}.
Determine if the primary sentiment expressed is positive, negative, or neutral.
Provide a brief explanation (1-2 sentences) focusing on the key words or phrases that justify your classification.

Respond ONLY with the following two lines:
Bearing: [positive/negative/neutral]
Explanation: [Your brief explanation]
"""

        examples_str = ""
        if technique == "few_shot":
            examples = self.bearing_examples.get(language.lower(), self.bearing_examples["english"])
            examples_str = "\nHere are some examples:\n"
            for ex in examples[:2]: # Limit examples
                if 'text' in ex and 'bearing' in ex and 'explanation' in ex:
                    examples_str += f'\nText: "{ex["text"]}"\nBearing: {ex["bearing"]}\nExplanation: {ex["explanation"]}\n'
            examples_str += "\nNow analyze the new text:\n"

        prompt = f"""{instruction}
{examples_str}
--- TEXT TO ANALYZE ---
"{text}"
--- END TEXT ---

ANALYSIS:
"""
        return prompt

    def _parse_bearing_response(self, response_raw: str) -> Dict[str, Any]:
        """Parses the LLM response to extract bearing and explanation."""
        result = {
            'bearing': 'neutral', # Default to neutral if unsure
            'explanation': 'Could not parse explanation.',
            'is_valid': False,
            'error_message': None
        }
        response_lower = response_raw.lower()

        try:
            bearing = 'neutral' # Default
            # Prioritize explicit matches
            bearing_match = re.search(r'bearing:\s*(positive|negative|neutral)', response_lower)
            if bearing_match:
                 bearing = bearing_match.group(1)
                 result['is_valid'] = True # Found explicit bearing
            # Fallback: check keywords if explicit match failed
            elif 'positive' in response_lower: bearing = 'positive'
            elif 'negative' in response_lower: bearing = 'negative'
            result['bearing'] = bearing

            # Explanation
            exp_match = re.search(r'explanation:\s*(.*)', response_raw, re.IGNORECASE | re.DOTALL)
            if exp_match:
                 explanation_full = exp_match.group(1).strip()
                 result['explanation'] = explanation_full.split('\n')[0][:300] # Limit length
                 result['is_valid'] = True # Got an explanation
            elif result['is_valid']: # If bearing was found but no explanation line
                 result['explanation'] = "Explanation line not found in response."
            else: # If neither bearing nor explanation found
                 result['explanation'] = "Could not parse bearing or explanation from response."
                 result['error_message'] = "Invalid response format"


        except Exception as e:
            print(f"Error parsing LLM response: {e}\nResponse: {response_raw[:500]}")
            result['explanation'] = f"Parsing error: {e}"
            result['is_valid'] = False
            result['error_message'] = f"Parsing Exception: {e}"

        return result

    def classify_bearing(self, text: str, language: str, prompt_technique: str) -> Dict[str, Any]:
        """Classifies sentiment bearing for a single piece of text."""
        if not isinstance(text, str) or not text.strip():
            return {'bearing': 'neutral', 'explanation': 'Input text is empty.', 'is_valid': False}

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {'bearing': 'neutral', 'explanation': 'Text filtered out by cleaning.', 'is_valid': False}

        prompt = self._create_bearing_prompt(cleaned_text, language, prompt_technique)

        try:
            start_time = time.time()
            response_raw = self.llm.generate(prompt)
            end_time = time.time()
            print(f"LLM call took {end_time - start_time:.2f}s for text: '{cleaned_text[:50]}...'")

            if not isinstance(response_raw, str): response_raw = str(response_raw)

            parsed_result = self._parse_bearing_response(response_raw)
            # Add original/cleaned text for context if needed later
            # parsed_result['analyzed_text'] = cleaned_text
            return parsed_result

        except Exception as e:
            print(f"Error during LLM call for text '{cleaned_text[:50]}...': {e}")
            traceback.print_exc()
            return {
                'bearing': 'error',
                'explanation': f"LLM call failed: {e}",
                'is_valid': False,
                'error_message': str(e)
            }

    # --- Methods for processing files and lexicons ---

    def _get_text_chunks(self, text: str, max_chunk_chars: int = 1500) -> List[str]:
        """Splits large text into manageable chunks."""
        if len(text) <= max_chunk_chars:
            return [text]

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
                    except Exception as nltk_err:
                         print(f"NLTK sentence tokenization failed: {nltk_err}, using basic split.")
                         sentences = re.split(r'(?<=[.?!])\s+', para) # Basic split
                else:
                     sentences = re.split(r'(?<=[.?!])\s+', para) # Basic split

                temp_chunk_sentences = []
                temp_chunk_len = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent: continue
                    if temp_chunk_len > 0 and temp_chunk_len + len(sent) + 1 > max_chunk_chars:
                         chunk_to_add = " ".join(temp_chunk_sentences)
                         if self.clean_text(chunk_to_add): chunks.append(chunk_to_add) # Clean final chunk
                         temp_chunk_sentences = [sent]
                         temp_chunk_len = len(sent)
                    else:
                         temp_chunk_sentences.append(sent)
                         temp_chunk_len += len(sent) + 1
                # Add the last part of the sentence-split paragraph
                if temp_chunk_sentences:
                     chunk_to_add = " ".join(temp_chunk_sentences)
                     if self.clean_text(chunk_to_add): chunks.append(chunk_to_add)

            # If paragraph fits, add it to the current chunk or start new one
            else:
                if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if self.clean_text(current_chunk): chunks.append(current_chunk)
                    current_chunk = para

        # Add the very last chunk
        if self.clean_text(current_chunk): chunks.append(current_chunk)

        # Final check: if any chunk is still too large (e.g., no sentence breaks), split by length
        final_chunks = []
        for chunk in chunks:
             if len(chunk) > max_chunk_chars:
                  print(f"Warning: Chunk still too long after sentence/para split ({len(chunk)} chars). Force splitting.")
                  final_chunks.extend([chunk[i:i+max_chunk_chars] for i in range(0, len(chunk), max_chunk_chars)])
             elif chunk:
                  final_chunks.append(chunk)

        return final_chunks


    def classify_bootstrapped_lexicon(self, language: str, limit: Optional[int], prompt_technique: str) -> pd.DataFrame:
        """Reads and classifies terms from a bootstrapped lexicon file."""
        results = []
        lexicon_filename = f"{language}_lexicon_bootstrapped.csv"
        # Look in common locations
        paths_to_check = [
            lexicon_filename,
            os.path.join("data", lexicon_filename),
            os.path.join("..", "data", lexicon_filename)
        ]
        actual_path_used = None
        for path in paths_to_check:
            if os.path.exists(path):
                actual_path_used = path
                break

        if not actual_path_used:
            print(f"Error: Bootstrapped lexicon file not found at expected paths for language '{language}'. Checked: {paths_to_check}")
            return pd.DataFrame({"error": ["Lexicon file not found"]})

        print(f"Loading bootstrapped lexicon from: {actual_path_used}")
        try:
            df = pd.read_csv(actual_path_used)
            if df.empty: return pd.DataFrame({"message": ["Lexicon file is empty"]})

            # --- Identify Text Column ---
            text_col = None
            potential_cols = ['term', 'word', 'phrase', 'text', 'lemma', 'content']
            df_cols_lower = {col.lower(): col for col in df.columns}
            for col_potential in potential_cols:
               if col_potential in df_cols_lower: text_col = df_cols_lower[col_potential]; break
            if not text_col:
                if df.columns.any(): text_col = df.columns[0]
                else: return pd.DataFrame({"error": ["Lexicon CSV has no columns"]})
                print(f"Warning: Using first column '{text_col}' as text column for lexicon.")
            # --- End Text Column Identification ---

            df_to_process = df.head(limit) if limit and limit > 0 else df
            print(f"Classifying {len(df_to_process)} terms from lexicon...")

            for idx, row in df_to_process.iterrows():
                term = str(row[text_col]) if pd.notna(row[text_col]) else ""
                result_data = {text_col: term} # Start with original term

                if term.strip():
                    classification_result = self.classify_bearing(term, language, prompt_technique)
                    result_data.update(classification_result) # Add bearing, explanation, etc.
                else:
                    result_data.update({'bearing': 'neutral', 'explanation': 'Empty term in lexicon.', 'is_valid': False})

                results.append(result_data)
                # Optional delay for rate limits
                # time.sleep(0.05)

            print("Lexicon classification finished.")
            return pd.DataFrame(results)

        except Exception as e:
             print(f"Error processing lexicon file {actual_path_used}: {e}")
             traceback.print_exc()
             return pd.DataFrame({"error": [f"Error processing lexicon: {e}"]})


    def classify_uploaded_file(self, uploaded_file, language: str, prompt_technique: str) -> pd.DataFrame:
        """Classifies content from an uploaded file (PDF, TXT, CSV)."""
        results = []
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()
        print(f"Processing uploaded file: {file_name} (Type: {ext})")

        try:
            if ext == '.pdf':
                if not PYMUPDF_INSTALLED:
                    return pd.DataFrame({"error": ["PyMuPDF not installed, cannot process PDF."]})

                full_text = ""
                page_texts = {} # Store text per page for chunk metadata
                try:
                    # Read PDF content using PyMuPDF
                    pdf_doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                    print(f"Opened PDF with {len(pdf_doc)} pages.")
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc.load_page(page_num)
                        page_text = page.get_text("text", sort=True).strip() # Get sorted text
                        cleaned_page_text = self.clean_text(page_text) # Clean page text
                        if cleaned_page_text:
                             page_texts[page_num + 1] = cleaned_page_text
                             full_text += cleaned_page_text + "\n\n" # Combine cleaned text
                    pdf_doc.close()
                except Exception as pdf_err:
                     print(f"Error reading PDF content: {pdf_err}")
                     traceback.print_exc()
                     return pd.DataFrame({"error": [f"PDF Reading Error: {pdf_err}"]})

                if not page_texts:
                    return pd.DataFrame({"message": ["No processable text extracted from PDF after cleaning."]})

                # Chunk the text from pages (could also chunk full_text)
                all_chunks_with_page = []
                for page_num, text in page_texts.items():
                     chunks = self._get_text_chunks(text) # Chunk cleaned page text
                     for chunk in chunks:
                          all_chunks_with_page.append({"text": chunk, "page": page_num})

                print(f"Generated {len(all_chunks_with_page)} text chunks from PDF.")
                # Process chunks
                for i, item in enumerate(all_chunks_with_page):
                     print(f"  Classifying PDF chunk {i+1}/{len(all_chunks_with_page)} (Page: {item['page']})...")
                     chunk_text = item["text"]
                     result_data = {"text": chunk_text, "page": item["page"]}
                     classification_result = self.classify_bearing(chunk_text, language, prompt_technique)
                     result_data.update(classification_result)
                     results.append(result_data)
                     # time.sleep(0.05) # Optional delay

            elif ext == '.txt':
                try:
                    content_bytes = uploaded_file.getvalue()
                    try: content = content_bytes.decode('utf-8')
                    except UnicodeDecodeError: content = content_bytes.decode('latin-1')
                except Exception as read_err:
                     return pd.DataFrame({"error": [f"TXT Reading Error: {read_err}"]})

                cleaned_content = self.clean_text(content)
                if not cleaned_content:
                     return pd.DataFrame({"message": ["No processable text in TXT after cleaning."]})

                chunks = self._get_text_chunks(cleaned_content)
                print(f"Generated {len(chunks)} text chunks from TXT.")
                # Process chunks
                for i, chunk_text in enumerate(chunks):
                    print(f"  Classifying TXT chunk {i+1}/{len(chunks)}...")
                    result_data = {"text": chunk_text} # No page number for TXT
                    classification_result = self.classify_bearing(chunk_text, language, prompt_technique)
                    result_data.update(classification_result)
                    results.append(result_data)
                    # time.sleep(0.05) # Optional delay

            elif ext == '.csv':
                try:
                    bytes_data = uploaded_file.getvalue()
                    try: s = bytes_data.decode('utf-8')
                    except UnicodeDecodeError: s = bytes_data.decode('latin-1')
                    data = io.StringIO(s)
                    df = pd.read_csv(data)
                except Exception as read_err:
                     return pd.DataFrame({"error": [f"CSV Reading/Parsing Error: {read_err}"]})

                if df.empty: return pd.DataFrame({"message": ["CSV file is empty"]})

                # --- Identify Text Column ---
                text_col = None
                potential_cols = ['text', 'content', 'sentence', 'comment', 'review', 'phrase', 'word', 'term']
                df_cols_lower = {col.lower(): col for col in df.columns}
                for col_potential in potential_cols:
                   if col_potential in df_cols_lower: text_col = df_cols_lower[col_potential]; break
                if not text_col:
                    if df.columns.any(): text_col = df.columns[0]
                    else: return pd.DataFrame({"error": ["CSV has no columns"]})
                    print(f"Warning: Using first column '{text_col}' as text column for CSV.")
                # --- End Text Column Identification ---

                print(f"Classifying {len(df)} rows from CSV (Column: '{text_col}')...")
                for idx, row in df.iterrows():
                     print(f"  Classifying CSV row {idx+1}/{len(df)}...")
                     original_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                     result_data = {"text": original_text} # Keep original

                     if original_text.strip():
                          classification_result = self.classify_bearing(original_text, language, prompt_technique)
                          result_data.update(classification_result)
                     else:
                          result_data.update({'bearing': 'neutral', 'explanation': 'Empty text in row.', 'is_valid': False})

                     # Add other columns from original CSV for context? Optional.
                     # for col in df.columns:
                     #     if col != text_col: result_data[f"csv_{col}"] = row[col]

                     results.append(result_data)
                     # time.sleep(0.05) # Optional delay

            else:
                return pd.DataFrame({"error": [f"Unsupported file type: '{ext}'. Use PDF, TXT, or CSV."]})

            print("File classification finished.")
            return pd.DataFrame(results)

        except Exception as e:
            print(f"An unexpected error occurred processing file '{file_name}': {e}")
            traceback.print_exc()
            return pd.DataFrame({"error": [f"Unexpected Error: {e}"]})
