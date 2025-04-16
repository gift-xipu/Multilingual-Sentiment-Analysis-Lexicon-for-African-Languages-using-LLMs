# Multilingual Sentiment Analysis Lexicon for African Languages using LLMS

## Introduction

This is a code snippet of the research done by Gift Markus Xipu, from the University of Johannesburg

## How it works:

1. We used four LLMs, Claude, OpenAI, Gemini, and Deepseek v1-3. (More models could be used but due to hardware restraints we stuck to these)
2. The Languages tested here were in Sesotho, Sepedi and Setswana.
3. We also used Zero-shot and Few-shot Prompting to test the different results and see which one is effective 
4. This system not only gives sentiment bearings and classifications but gives explanations of how the model got there

## Sentiment Approaches

1. Prompt-Based Lexicon Generation:
    - Uses prompt engineering with LLMs to generate sentiment scores for words/phrases.
    - Allows creation of domain-specific and multilingual lexicons dynamically.
2. Self-Explaining Lexicons via LLM Rationales:
    - Asks LLMs to explain sentiment, not just score words.
    - Adds rationales to lexicon entries for better explainability.
3. Lexicon Bootstrapping with Synthetic Data + Human-in-the-Loop:
    - LLMs generate synthetic sentences with a target word.
    - Sentence sentiment is used to infer the word's sentiment, creating context-aware lexicons.

## Tech Stack:

1. Python
2. Streamlit
