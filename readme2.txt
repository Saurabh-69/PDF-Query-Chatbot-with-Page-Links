**********************************README FILE FOR final6_code_main.py**********************************


PDF Query Chatbot with Page Links
Overview
This project is a PDF query chatbot that allows users to enter queries and extract similar or closely matching data from two specified PDF files. The chatbot uses natural language processing to find relevant text and provides links to specific pages within the PDFs for easy reference.

Features
Text Extraction: Extracts and cleans text from two PDF files using the PyMuPDF library.
Query Suggestion: Suggests similar queries based on previously stored queries using Python's difflib.
Similarity Calculation: Calculates text similarity between user queries and extracted PDF text using TF-IDF and cosine similarity.
Dynamic UI: Built with Gradio for an interactive user experience.
PDF Page Linking: Generates clickable links to specific pages in the PDFs for quick access.
Requirements
Before running the application, ensure you have the following installed:

Python 3.x

Required libraries (install via pip):

bash
Copy code
pip install PyMuPDF gradio google-cloud-dialogflow google-cloud-storage scikit-learn
Google Cloud Credentials: Set up Google Cloud Dialogflow and obtain the necessary service account JSON file. Update the credentials path in the ChatWrapper class.

Project Structure
app.py: Main application file containing the chatbot logic.
stored_queries.txt: A text file to store previously entered queries for suggestion purposes.
pdf1.pdf & pdf2.pdf: Sample PDF files to be processed (ensure these files exist in the specified paths).
Setup
Prepare PDF Files: Place the PDF files you want to analyze in the specified path or update the paths in the pdf_paths dictionary within app.py.

Configure Google Cloud: Ensure your Google Cloud credentials file path is correctly specified in the ChatWrapper class.

Run the Application: Execute the following command to start the Gradio interface:

bash
Copy code
python app.py
Access the Interface: Open the provided link in your web browser to interact with the chatbot.

How to Use
Enter your query in the input box.
The chatbot will suggest closely matching queries if available.
Upon submitting a query, the chatbot will display:
Matched text from the PDFs.
The most relevant page number.
The name of the PDF.
A button to open the specific page in the PDF.
A direct link to the page in the PDF.
Click on the "Open PDF Page" button to view the relevant page in Google Chrome.
Code Walkthrough
Text Cleaning: The clean_text function removes unnecessary patterns from the extracted text.
Text Extraction: The extract_text_from_pdfs function reads and processes text from specified PDFs.
Dialogflow Integration: The ChatWrapper class handles interaction with Dialogflow for natural language processing.
Similarity Calculation: The find_most_relevant_pages function computes similarity scores to identify the most relevant text.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
PyMuPDF for PDF text extraction.
Gradio for the user interface.
Google Cloud Dialogflow for natural language understanding.
Scikit-learn for machine learning utilities.

**************************************************************************************************************************************************

Key Concepts and Components
PDF Text Extraction:

The project uses the PyMuPDF library to read and extract text from PDF files. This is essential for making the contents of the PDFs searchable and retrievable based on user queries.
Natural Language Processing (NLP):

The chatbot leverages Google Dialogflow, a cloud-based conversational platform, to interpret user queries. Dialogflow utilizes machine learning to understand the intent behind the user's input, enabling more natural interactions.
Cosine Similarity:

The project employs the concept of cosine similarity to measure the similarity between the user query and the extracted text from PDFs. This mathematical approach helps determine how closely related two pieces of text are, allowing the chatbot to return the most relevant sections of the PDFs.
TF-IDF Vectorization:

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text into numerical vectors. This method helps identify important words in the text, emphasizing unique words while downplaying common terms, thus improving the accuracy of similarity measurements.
Gradio Interface:

Gradio is utilized to create an interactive user interface for the chatbot. It allows users to input their queries, receive suggestions, and view results in a user-friendly format. Gradio simplifies the process of building web-based applications, making it accessible for users without extensive web development knowledge.
Query Suggestions:

The project includes a feature to suggest similar queries based on previously stored queries. This is accomplished using the difflib library, which provides close matches to user input, enhancing user experience by guiding them toward relevant queries.
Retry Mechanism:

The chatbot incorporates a retry mechanism when interacting with Dialogflow. If the response from Dialogflow is not satisfactory or lacks relevant information, the chatbot can re-send the query to obtain better results. This ensures a more reliable interaction.
File URLs for PDF Pages:

The application generates clickable URLs that direct users to specific pages in the PDF documents. This functionality enhances user convenience, allowing them to access the exact information they are looking for quickly.
Session Management:

Each user session is assigned a unique identifier using UUIDs, ensuring that interactions remain distinct and allowing for organized tracking of user queries and responses.
Storage of Queries:

The project includes functions for loading and saving user queries to a text file. This feature enables the application to retain user interactions over time, providing a historical context for suggestions and improving future user experiences.
Summary
Overall, the project combines several advanced techniques and tools to create a robust PDF query chatbot. By integrating PDF text extraction, natural language processing, and user interface design, it provides a seamless way for users to interact with document contents, making information retrieval efficient and user-friendly. This approach enhances the accessibility of information stored in PDFs, which can be cumbersome to navigate manually.