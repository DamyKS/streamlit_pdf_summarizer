# 📄 PDF Text Extractor & Summarizer

A Streamlit application that allows users to upload a PDF document and automatically extract its text, then summarize the content using Google's Gemini AI.

## ✨ Features

* **PDF Text Extraction:** Upload PDF files to extract all readable text content.
* **AI-Powered Summarization:** Get concise summaries of your PDF text, generated by Google's Gemini AI.
* **Adjustable Summary Length:** Choose between short, medium, and long summaries to fit your needs.
* **Progress Indicators:** Clear visual feedback during text extraction and summarization.
* **User-Friendly Interface:** Simple and intuitive design for easy navigation and use.
  
Demo Link : https://streamlit-pdf-summarizer.onrender.com

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your system:

* **Python 3.8+**
* **pip** (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DamyKS/streamlit_pdf_summarizer.git
    cd streamlit_pdf_summarizer
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key:**
    * Obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create a file named `.env` in the root of your `streamlit_pdf_summarizer` project directory (same level as `app.py`).
    * Add your API key to the `.env` file:
        ```
        GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```
    * **Important:** Make sure to add `/ .env` to your `.gitignore` file to prevent your API key from being committed to your public repository.

### Running the App

1.  **Make sure your virtual environment is active.**
2.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The app will typically open automatically in your web browser at `http://localhost:8501`.

## 💡 Usage Guidelines

1.  **Upload PDF:** Use the "Choose a PDF file" button to select your document.
2.  **Text Extraction:** The app will automatically extract text from the PDF.
3.  **View Extracted Text:** An expander will appear allowing you to view the full extracted text.
4.  **Choose Summary Length:** Select "Short," "Medium," or "Long" to get a summary of your desired length.
5.  **View Summary:** The generated summary will appear below the length options.

## 📄 License

This project is open-source and available under the MIT License

Built by **Damian**
