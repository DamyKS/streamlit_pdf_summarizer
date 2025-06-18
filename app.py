import streamlit as st
from pypdf import PdfReader
import io
import google.generativeai as genai
from decouple import config

# Configure the Google Generative AI with your API key
genai.configure(api_key=config("GOOGLE_API_KEY"))

# Define constants for summary lengths
SHORT_WORDS = 300
MEDIUM_WORDS = 500
LONG_WORDS = 1000


# --- Helper Function for Text Summarization ---
def get_gemini_summary(text_content: str, max_words: int) -> str:
    """
    Generates a summary of the given text content using the Gemini AI model.
    """
    if not text_content:
        return "No text provided for summarization."

    prompt = (
        f"Summarize the following text concisely, focusing on key information and main points. "
        f"The summary should be approximately {max_words} words. "
        f"Crucially, **DO NOT include any introductory phrases, conversational filler, or self-references like 'This text provides' or 'Here is a summary'.** "
        f"Provide ONLY the summary content.\n\n"
        f"Text to summarize:\n{text_content}"
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)

        if response and response.candidates:
            if response.candidates[0].content and response.candidates[0].content.parts:
                summary = response.candidates[0].content.parts[0].text
                return summary
            else:
                st.error("Gemini model response content is empty or malformed.")
                return (
                    "Failed to get summary: Empty or malformed response from AI model."
                )
        else:
            st.error("Gemini model did not return any candidates.")
            return "Failed to get summary: No candidates from AI model."
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return f"An error occurred during summarization: {e}"


# extract and cache pdf  content
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "".join([page.extract_text() or "" for page in reader.pages])


# --- Streamlit App Layout ---
st.set_page_config(page_title="PDF Summarizer", layout="centered")

st.title("📄 PDF Text Extractor & Summarizer")
st.markdown("Upload a PDF and get a quick summary powered by Google Gemini AI.")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for summary length if not already set
if "summary_length_option" not in st.session_state:
    st.session_state.summary_length_option = "Medium"  # Default selection

text_content = ""
summary = ""

if uploaded_file is not None:
    # Display a spinner while processing PDF extraction
    with st.spinner("Extracting text from PDF..."):
        try:

            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                text_content = extract_text_from_pdf(file_bytes)

            if not text_content.strip():
                st.warning("No readable text could be extracted from the PDF.")
                # Clear existing summary if no text is extracted
                st.session_state.current_summary = ""
            else:
                st.success("Text extracted successfully!")

                # Expander for extracted text
                with st.expander("Extracted Text"):
                    st.text_area(
                        "Full text from PDF", text_content, height=300, disabled=True
                    )

                # Summarize the text if extraction was successful
                if text_content.strip():
                    # Create two columns for "Summary" header and length options
                    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

                    with col1:
                        st.subheader("Summary")

                    with col2:
                        st.write("Summary Length:")
                        # Radio buttons for summary length
                        st.session_state.summary_length_option = st.radio(
                            "Choose length",
                            ("Short", "Medium", "Long"),
                            index=("Short", "Medium", "Long").index(
                                st.session_state.summary_length_option
                            ),
                            horizontal=True,  # Display buttons horizontally
                            label_visibility="collapsed",  # Hide the default label as we have "Summary Length:" text
                        )

                    # Determine max_words based on selected option
                    current_max_words = MEDIUM_WORDS  # Default if somehow not set
                    if st.session_state.summary_length_option == "Short":
                        current_max_words = SHORT_WORDS
                    elif st.session_state.summary_length_option == "Medium":
                        current_max_words = MEDIUM_WORDS
                    elif st.session_state.summary_length_option == "Long":
                        current_max_words = LONG_WORDS

                    # Store text_content in session state to avoid re-extracting on every rerun due to button click
                    st.session_state.extracted_text = text_content

                    # Only generate summary if there's extracted text and a file is uploaded
                    if st.session_state.extracted_text and uploaded_file is not None:
                        with st.spinner(
                            f"Generating {st.session_state.summary_length_option} summary with Gemini AI..."
                        ):
                            summary = get_gemini_summary(
                                st.session_state.extracted_text, current_max_words
                            )
                            if summary:
                                # Store the summary in session state to persist across reruns
                                st.session_state.current_summary = summary
                            else:
                                st.session_state.current_summary = (
                                    "Could not generate summary."
                                )

                    # Display the current summary (either newly generated or from session state)
                    st.write(st.session_state.current_summary)

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.warning("Please ensure you've uploaded a valid and readable PDF file.")
            # Clear summary on error
            st.session_state.current_summary = ""

# Ensure summary display persists even if file_uploader becomes None on rerun, if a summary exists
if (
    "current_summary" in st.session_state
    and st.session_state.current_summary
    and uploaded_file is None
):
    st.subheader("Summary")  # Re-display header if needed
    st.write(st.session_state.current_summary)


st.markdown("---")
st.markdown("Built by **Damian**")
