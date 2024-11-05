# PDF-Query-Chatbot-with-Page-Links

PDF Query Chatbot with Page Links
This project is a chatbot interface for querying and extracting relevant information from two specified PDF documents. It uses NLP and machine learning to analyze, match, and retrieve text data based on user input. With the integration of Google Dialogflow for natural language understanding and TF-IDF for text similarity, the chatbot provides answers and links directly to the relevant page within the PDFs.

Features
Text Matching: Extracts relevant text and finds closely matching content between two PDF documents.
Query Suggestions: Suggests relevant stored queries as the user types.
Relevant Page Link: Displays the PDF path, relevant page number, and a button to open the page in Chrome.
Gradio Interface: Provides an interactive UI to enter questions, view matched content, and open the PDF link.
Requirements
Python 3.8 or higher

Required Python Libraries:

fitz (PyMuPDF): PDF handling
gradio: UI framework
google-cloud-dialogflowcx: For integrating Google Dialogflow
google-auth: For Google service authentication
sklearn: For TF-IDF and cosine similarity calculations
webbrowser: For opening the PDF page in Chrome
Google Cloud setup for Dialogflow CX with dialogflowcx_v3beta1 and a service account JSON file.

Two PDFs for comparison.

Chrome installed on the system for opening page links.

Setup Instructions
Clone the repository and navigate to the project directory.
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Place your service account JSON file in the project directory and update its path in the code:
python
Copy code
credentials = Credentials.from_service_account_file("path_to_your_json_file.json")
Specify the path to your PDF files in the pdf_paths dictionary:
python
Copy code
pdf_paths = {
    'pdf1': 'path_to_pdf1.pdf',
    'pdf2': 'path_to_pdf2.pdf'
}
Set the path to Chrome in chrome_path to open PDF pages:
python
Copy code
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
Run the application using:
bash
Copy code
python app.py
How It Works
User Query: Enter a query through the Gradio interface.
Query Processing:
Google Dialogflow interprets the query for relevant text.
TF-IDF and cosine similarity scores identify the most relevant content in the PDFs.
Output:
Displays the matching text, page number, PDF name, and a button to open the PDF page.
Provides a URL for manual access to the PDF page.
Code Overview
Text Cleaning: clean_text() cleans extracted text from PDFs.
Stored Queries: Functions load_stored_queries() and save_stored_queries() manage query storage.
TF-IDF Vectorization: create_tfidf_vectorizer() uses TF-IDF to measure content similarity.
Google Dialogflow Integration: ChatWrapper connects to Dialogflow, with retry_dialogflow_response() to handle retries.
Text Extraction from PDF: extract_text_from_pdfs() extracts text from specified PDFs.
Query Matching and Retrieval: query_pdf_gemini() retrieves the most relevant text and page.
UI with Gradio: Provides an interactive UI for querying and viewing results.
Usage
Enter a query in the text input field.
View suggestions and relevant matched text.
Click the "Open PDF Page" button to view the corresponding page in Chrome.
Troubleshooting
PDF not opening: Ensure Chrome is installed and chrome_path is correct.
Dialogflow issues: Check service account permissions and project configurations.
No results found: Ensure PDFs are correctly loaded and have searchable text.
