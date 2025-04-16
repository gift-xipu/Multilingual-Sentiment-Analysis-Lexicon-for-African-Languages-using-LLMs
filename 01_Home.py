import streamlit as st
from utils.shared_ui import ensure_session_state, display_sidebar, initialize_llm

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="African Languages Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure session state variables are initialized
ensure_session_state()

# Display sidebar and get config selections
model_choice, language_choice = display_sidebar()

# --- Home Page Content ---
st.title("Multilingual Sentiment Analysis for African Languages")
st.subheader("Leveraging Large Language Models for Low-Resource African Languages")

# Main description with enhanced formatting
st.markdown("""
<div style='background-color: rgba(100, 65, 165, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
<h3 style='margin-top: 0;'>Research Overview</h3>
<p>This application supports research into sentiment analysis for low-resource African languages 
(Sesotho, Sepedi, and Setswana) using Large Language Models (LLMs). The project explores how LLMs can be leveraged 
to develop sentiment lexicons and perform effective sentiment analysis without translation to high-resource languages.</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Configuration:")
    # Display current selections from session state
    st.write(f"- **Selected Model:** `{st.session_state.get('current_model_name', 'N/A')}`")
    st.write(f"- **Selected Language:** `{st.session_state.get('current_language', 'N/A')}`")
    
    # Check if model is initialized
    if st.session_state.get('current_model'):
        st.success(f"✅ {st.session_state.current_model_name} model is initialized and ready.")
    else:
        st.error(f"⚠️ {st.session_state.current_model_name} model is not initialized. Please provide the required API key in the sidebar.")

with col2:
    st.subheader("Application Features")
    st.markdown("""
    - **📚 Generating Lexicons**: Create sentiment lexicons for African languages
    - **🔍 Sentiment Bearings**: Identify sentiment polarity in words and expressions
    - **📊 Sentiment Classification**: Categorize text based on sentiment analysis
    - **🔄 Lexicon Bootstrapping**: Expand lexicons through automated bootstrapping techniques
    """)

# Add a separator
st.markdown("---")

# Research objectives
st.subheader("Research Objectives")
st.markdown("""
This research project addresses key gaps in sentiment analysis for African languages:

1. **Evaluate LLM Capabilities**: Assess the performance of large language models in sentiment analysis 
   tasks directly on African languages without translation
   
2. **Explore Prompting Strategies**: Investigate different prompting techniques (zero-shot and few-shot learning) 
   to optimize sentiment analysis for African languages
   
3. **Create Sentiment Lexicons**: Develop language-specific sentiment lexicons for Sesotho, Sepedi, and Setswana
   that capture cultural and linguistic nuances
   
4. **Compare Model Architectures**: Identify the most suitable LLM approaches for African language sentiment analysis
""")

# Problem statement section
st.markdown("---")
st.subheader("The Challenge")
st.markdown("""
While sentiment analysis has been extensively studied for high-resource languages like English, 
African languages remain significantly underrepresented due to:

- Scarcity of linguistic resources and specialized models
- Limitations of translation-based approaches that fail to capture cultural and linguistic nuances
- Lack of comprehensive sentiment lexicons for African languages
- Limited research on applying recent LLM advances to African language contexts

This application seeks to address these challenges by providing tools to develop and evaluate
sentiment analysis approaches specifically designed for African languages.
""")

# How to use section
st.markdown("---")
st.subheader("Getting Started")
st.markdown("""
1. Select your preferred LLM and target language in the sidebar
2. Provide any required API keys for model access
3. Navigate to the specific tool you wish to use through the sidebar menu
4. Follow the instructions on each page to conduct sentiment analysis or generate lexicons

Our research aims to demonstrate that LLMs can effectively analyze sentiment in African languages
without relying on translation to high-resource languages, while capturing the cultural and
linguistic nuances specific to each language.
""")

# Footer with attribution
st.markdown("---")
st.markdown("<div style='text-align: center;'><p><strong>African Languages Sentiment Analysis Research</strong> • Gift Markus Xipu • 2025</p></div>", unsafe_allow_html=True)
