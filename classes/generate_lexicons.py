import pandas as pd
import os

class GenerateLexicon:
    def __init__(self, llm_model, language, prompt_technique='zero-shot'):
        """
        Initializes the lexicon generator.

        Args:
            llm_model: An instance of a language model interface with a 'generate' method.
            language (str): The target language (e.g., 'sesotho', 'english').
            prompt_technique (str): Prompting technique ('zero-shot' or 'few-shot'). Defaults to 'zero-shot'.
        """
        self.llm_model = llm_model
        self.language = language.lower()
        self.prompt_technique = prompt_technique.lower()
        self.base_data_path = "data" # Base directory for data files
        self.csv_path = os.path.join(self.base_data_path, f"{self.language}.csv")
        self.columns = ["word", "meaning"] # Expected columns in the CSV

        # Few-shot examples for different languages (focused on sentiment)
        self.few_shot_examples = {
            "sesotho": [
                {"word": "thabile", "meaning": "Happy or joyful"},
                {"word": "bohloko", "meaning": "Pain or suffering"},
                {"word": "halefile", "meaning": "Angry or furious"},
                {"word": "monate", "meaning": "Nice, pleasant or delicious"},
                {"word": "khathetse", "meaning": "Tired or weary"},
            ],
            "sepedi": [
                {"word": "thabišwa", "meaning": "Happy or pleased"},
                {"word": "bohloko", "meaning": "Pain or hurt"},
                {"word": "befetšwe", "meaning": "Very angry or enraged"},
                 {"word": "bose", "meaning": "Goodness, beauty, pleasantness"},
                 {"word": "lapišago", "meaning": "Tiring"},
            ],
            "setswana": [
                {"word": "itumetse", "meaning": "Joyful or content"},
                {"word": "botlhoko", "meaning": "Painful or sad"},
                {"word": "tenegile", "meaning": "Strongly annoyed or angered"},
                {"word": "monate", "meaning": "Nice, pleasant"},
                {"word": "lapile", "meaning": "Tired"},
            ],
            "english": [
                {"word": "happy", "meaning": "Feeling or showing pleasure or contentment."},
                {"word": "sad", "meaning": "Feeling or showing sorrow; unhappy."},
                {"word": "angry", "meaning": "Feeling or showing strong annoyance, displeasure, or hostility."},
                {"word": "joyful", "meaning": "Feeling, expressing, or causing great pleasure and happiness."},
                {"word": "tired", "meaning": "Feeling in need of rest or sleep."}
            ]
        }

    def load_existing_lexicon(self):
        """Loads the existing lexicon from the CSV file for the specified language."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8')
                # Ensure both expected columns exist, add if missing
                for col in self.columns:
                    if col not in df.columns:
                        print(f"Warning: Column '{col}' missing in {self.csv_path}. Adding it.")
                        df[col] = None # Add missing column, initialized to None
                # Select and reorder columns to ensure consistency
                return df[self.columns]
            except pd.errors.EmptyDataError:
                 print(f"Warning: Lexicon file {self.csv_path} is empty. Starting fresh.")
                 return pd.DataFrame(columns=self.columns)
            except Exception as e:
                print(f"Error loading lexicon from {self.csv_path}: {e}")
                # Return empty DataFrame on error to allow generation to proceed
                return pd.DataFrame(columns=self.columns)
        # Return empty DataFrame if file doesn't exist
        return pd.DataFrame(columns=self.columns)

    def generate_lexicon(self, categories=None):
        """
        Generates new sentiment lexicon entries using the LLM, adds them to the
        existing lexicon, and saves the result. Always attempts to generate 50 words.

        Args:
            categories (list, optional): A list of category strings to focus on. Defaults to None.

        Returns:
            pandas.DataFrame: The combined DataFrame including existing and newly generated entries.
        """
        existing_df = self.load_existing_lexicon()
        # Create a set of existing words (lowercase) for quick checking, handle potential NaN values
        existing_words = set(existing_df['word'].dropna().astype(str).str.lower())

        print(f"Starting lexicon generation for '{self.language}'.")
        print(f"Existing words loaded: {len(existing_words)}")

        attempts = 0
        max_attempts = 1 # Number of LLM calls (can increase if LLM often fails)
        new_entries = []
        num_words_target = 50  # Fixed target number of words to generate per run

        while len(new_entries) < num_words_target and attempts < max_attempts:
            # Calculate how many more words we need in this attempt
            remaining_words_needed = num_words_target - len(new_entries)
            print(f"\nAttempt {attempts + 1}/{max_attempts}: Requesting {remaining_words_needed} words...")

            # Create the prompt dynamically
            prompt = self._create_prompt(remaining_words_needed, categories, existing_words)

            try:
                # Call the LLM to generate words
                response = self.llm_model.generate(prompt)

                # Parse the LLM's response
                current_batch_df = self._parse_response(response, existing_words)
                print(f"Attempt {attempts + 1}: Parsed {len(current_batch_df)} potential new entries.")

                # Add valid, unique new entries from the batch
                added_count = 0
                for _, row in current_batch_df.iterrows():
                    word_lower = str(row['word']).lower() # Ensure string conversion and lowercasing
                    # Double check uniqueness against existing_words AND words added in this run
                    if word_lower not in existing_words:
                        new_entries.append(row.to_dict()) # Append as dictionary
                        existing_words.add(word_lower) # Add to set immediately to prevent duplicates within the same batch
                        added_count += 1
                        if len(new_entries) >= num_words_target:
                            break # Stop if we reached the target

                print(f"Attempt {attempts + 1}: Added {added_count} unique new words.")

            except Exception as e:
                print(f"Error during LLM generation or parsing on attempt {attempts + 1}: {e}")
                # Optionally, implement retry logic or error handling here

            attempts += 1

        print(f"\nGeneration finished. Total new unique words collected: {len(new_entries)}")

        # Combine and save if new entries were generated
        if new_entries:
            # Ensure new_entries is a list of dicts before creating DataFrame
            new_entries_df = pd.DataFrame(new_entries, columns=self.columns)

            # Concatenate old and new DataFrames
            # Ensure columns are aligned, especially if existing_df was empty or had missing cols
            combined_df = pd.concat([existing_df, new_entries_df], ignore_index=True)

            # Optional: Remove duplicates based on the 'word' column (case-insensitive)
            # Keep the first occurrence
            combined_df['word_lower'] = combined_df['word'].astype(str).str.lower()
            combined_df = combined_df.drop_duplicates(subset='word_lower', keep='first')
            combined_df = combined_df.drop(columns=['word_lower']) # Remove temporary column

            print(f"Total words in lexicon after adding new entries and deduplication: {len(combined_df)}")

            # Save the updated lexicon
            if self.save_lexicon(combined_df):
                 print(f"Successfully saved updated lexicon to {self.csv_path}")
            else:
                 print(f"Failed to save updated lexicon.")
            return combined_df
        else:
            print("No new words were added. Returning existing lexicon.")
            return existing_df # Return the original DataFrame if no new words were added

    def _create_prompt(self, num_words, categories, existing_words):
        """Creates the detailed prompt for the LLM, focusing on sentiment."""
        # --- Start of Improved Prompt ---

        prompt = f"""Your primary task is to generate EXACTLY {num_words} {self.language} words that distinctly express **sentiment** (positive, negative, or neutral feelings, opinions, or attitudes), along with their precise English meanings.

Context: These words are critical for sentiment analysis. Each word generated **must** carry a clear emotional charge or subjective evaluation.

Focus Areas:
* **Positive Sentiment:** Words indicating happiness, approval, satisfaction, excitement, etc.
* **Negative Sentiment:** Words indicating sadness, disapproval, anger, fear, disgust, etc.
* **Neutral Sentiment:** Words indicating indifference, objectivity, or states that are neither explicitly positive nor negative *but still relevant in a sentiment context* (e.g., 'surprising', 'unknown', 'standard' - use these sparingly compared to positive/negative).

**Avoid:**
* Purely descriptive words with no emotional/opinion content (e.g., 'blue', 'tall', 'table').
* Ambiguous words where sentiment is highly context-dependent without further information.
* Very long words or phrases; focus on single words or short, common idioms if applicable.

Categories: {f"Generate words specifically related to these themes: {', '.join(categories)}." if categories else "Generate general sentiment words across various common topics."}

Strict Output Rules:
1.  Produce exactly {num_words} unique word-meaning pairs.
2.  Format: `word,meaning` (exactly one comma separating the {self.language} word and its English meaning).
3.  Output ONLY the pairs, one pair per line.
4.  NO headers, titles, introductory text, explanations, or summaries in your response.
5.  NO markdown formatting (like `*` or ` ``` `).
6.  DO NOT include any of these already existing words (case-insensitive): {', '.join(list(existing_words)[:20]) if existing_words else 'None'}...
"""

        # Add few-shot examples if applicable, emphasizing their sentiment nature
        if self.prompt_technique == 'few-shot':
            # Fallback to English examples if specific language not found
            examples = self.few_shot_examples.get(self.language, self.few_shot_examples.get("english"))
            if examples:
                prompt += "\nExample Format (demonstrating sentiment words):\n"
                # Limit examples shown in prompt to avoid making it too long
                prompt += "\n".join(
                    [f"{ex['word']},{ex['meaning']}" for ex in examples[:3]] # Show only a few relevant examples
                )
                prompt += "\n" # Add a newline after examples

        # Final instruction emphasizing sentiment and rules
        prompt += f"""\nNow, generate exactly {num_words} new {self.language} **sentiment** words and their English meanings, strictly following all rules above."""

        # --- End of Improved Prompt ---
        return prompt


    def _parse_response(self, response, existing_words):
        """
        Parses the raw text response from the LLM into a DataFrame of words and meanings.

        Args:
            response (str): The raw text output from the LLM.
            existing_words (set): A set of lowercase existing words to avoid duplicates.

        Returns:
            pandas.DataFrame: A DataFrame containing the parsed new entries.
        """
        entries = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        for line in lines:
            # Skip lines that are likely comments, headers, or formatting artifacts
            if any(line.startswith(x) for x in ['#', '//', 'Example', '---', '```', 'word,meaning', 'Word,Meaning']):
                continue

            # Attempt to split by the first comma only
            parts = [p.strip().strip('"').strip("'") for p in line.split(',', 1)]

            if len(parts) == 2:
                word, meaning = parts
                word_lower = word.lower()

                # Basic validation: ensure word and meaning are not empty,
                # word is not already known (case-insensitive), and word is reasonably short.
                if (word and meaning and
                    word_lower not in existing_words and
                    len(word) > 0 and len(word) < 50): # Added len(word) > 0 check
                    entries.append({'word': word, 'meaning': meaning})
                # else:
                    # Optional: Print lines that were parsed but rejected
                    # print(f"  - Skipping line (duplicate or invalid): '{line}'")

        # Return DataFrame with specified columns
        return pd.DataFrame(entries, columns=self.columns)


    def save_lexicon(self, df):
        """
        Saves the given DataFrame to the language-specific CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            # Save only the specified columns, ensuring correct encoding
            df[self.columns].to_csv(self.csv_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error saving lexicon to {self.csv_path}: {e}")
            return False
