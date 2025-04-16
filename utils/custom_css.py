"""
Custom CSS utilities for consistent styling across all pages
of the African Languages Sentiment Analysis application.
"""

import streamlit as st

def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit application.
    Call this function after st.set_page_config() on each page.
    """
    st.markdown("""
    <style>
    /* Custom button styling to use secondaryBackgroundColor */
    .stButton button {
        background-color: #13102B !important;
        border: 1px solid #BB86FC !important;
        color: white !important;
    }
    
    /* Hover effect for buttons */
    .stButton button:hover {
        background-color: #1E1745 !important;
        border-color: #CF9FFF !important;
    }
    
    /* Pressed/active effect for buttons */
    .stButton button:active {
        background-color: #2A2156 !important;
    }
    
    /* Primary buttons (type="primary") styling */
    .stButton button[data-baseweb="button"] {
        background-color: #BB86FC !important;
        color: #0D0721 !important;
        font-weight: 600 !important;
    }
    
    /* Primary button hover state */
    .stButton button[data-baseweb="button"]:hover {
        background-color: #CF9FFF !important;
    }
    
    /* Custom styling for dataframes/tables */
    .stDataFrame {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Custom styling for text inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: #13102B !important;
        color: white !important;
        border: 1px solid #382C6C !important;
        border-radius: 5px !important;
    }
    
    /* Custom styling for selectboxes */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #13102B !important;
        border: 1px solid #382C6C !important;
    }
    
    /* Section headers with underline */
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        padding-bottom: 0.3rem !important;
        border-bottom: 2px solid #BB86FC !important;
        margin-bottom: 1rem !important;
    }
    
    /* Card-like containers */
    .custom-card {
        background-color: #13102B !important;
        border-radius: 10px !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 20px !important;
    }
    
    /* Success message styling */
    .element-container div[data-testid="stAlert"][kind="success"] {
        background-color: rgba(165, 214, 183, 0.2) !important;
        border-color: #A5D6B7 !important;
    }
    
    /* Info message styling */
    .element-container div[data-testid="stAlert"][kind="info"] {
        background-color: rgba(144, 202, 249, 0.2) !important;
        border-color: #90CAF9 !important;
    }
    
    /* Warning message styling */
    .element-container div[data-testid="stAlert"][kind="warning"] {
        background-color: rgba(255, 203, 107, 0.2) !important;
        border-color: #FFCB6B !important;
    }
    
    /* Error message styling */
    .element-container div[data-testid="stAlert"][kind="error"] {
        background-color: rgba(255, 83, 112, 0.2) !important;
        border-color: #FF5370 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def styled_container(content_function):
    """
    Creates a styled container with the custom-card CSS class.
    
    Args:
        content_function: A function that will be called to render content inside the container
    
    Example usage:
        with styled_container():
            st.write("This content is inside a styled container")
    """
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)

def styled_header(title):
    """
    Creates a styled section header with the section-header CSS class.
    
    Args:
        title: The header text to display
    
    Example usage:
        styled_header("Lexicon Generation")
    """
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)
