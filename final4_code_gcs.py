# THIS CODE IS DEVELOPED TO READ THE PDF FROM GOOGLE CLOUD OR GOOGLE BUCKETS ONLINE


# IMPORTING ALL THE NESECESSARY MOUDLES AND LIBRARIES
import fitz  # PyMuPDF
import gradio as gr
from google.cloud import dialogflowcx_v3beta1 as dialogflowcx
from google.oauth2.service_account import Credentials
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import difflib
import urllib.parse
import webbrowser
from google.cloud import storage
import os

# GCS CREDENTIALS
SERVICE_ACCOUNT_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# SPECIFYING THE PATH FOR CHROME EXECUTION
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))

# FUNCTION FOR TEXT CLEANING USING REGEX
def clean_text(extracted_text):
    patterns_to_remove = [
        r"file:///C:/Users/pravin\.sharma/AppData/Local/Temp/~hh\d+\.htm \d{1,2}/\d{1,2}/\d{4}",
        r"Acknowledge Review"
    ]
    for pattern in patterns_to_remove:
        extracted_text = re.sub(pattern, '', extracted_text)
    return extracted_text

#******************************** READING AND WRITING FROM TXT FILE *****************************************
#FUNTIONS TO READ AND WRITE THE STORED QUERIES TEXT FILE
def load_stored_queries(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def save_stored_queries(file_path, queries):
    with open(file_path, 'w') as file:
        for query in queries:
            file.write(query + '\n')

stored_queries = load_stored_queries(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\stored_queries.txt")


# FUNCTION TO SUGGEST QUERIES BASED ON USER INPUT
def get_query_suggestions(user_input, stored_queries):
    return difflib.get_close_matches(user_input, stored_queries, n=5, cutoff=0.3)


# FUNCTION TO CREATE TFIDF VECTOR
def create_tfidf_vectorizer():
    return TfidfVectorizer(stop_words='english', ngram_range=(1, 3))



# FUNCTION TO CREATE CHAT WRAPPER AND INTENT DETECTION
class ChatWrapper:
    def __init__(self):
        credentials = Credentials.from_service_account_file(
            r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\qpathways-dev-d8a2a251cb4f.json")
        session_id = str(uuid.uuid4())
        self.project_id = 'qpathways-dev'
        self.location = 'us-central1'
        self.agent_id = 'a37178c1-d1e6-4ed3-bb30-effc558cbfa2'
        self.session_id = session_id
        self.client = dialogflowcx.SessionsClient(
            credentials=credentials,
            client_options={'api_endpoint': f'{self.location}-dialogflow.googleapis.com'}
        )
        self.session_path = f"projects/{self.project_id}/locations/{self.location}/agents/{self.agent_id}/sessions/{self.session_id}"

    def detect_intent(self, user_message):
        query_input = dialogflowcx.QueryInput(
            text=dialogflowcx.TextInput(text=user_message),
            language_code="en-US"
        )
        request = dialogflowcx.DetectIntentRequest(
            session=self.session_path,
            query_input=query_input
        )
        response = self.client.detect_intent(request=request)
        return response.query_result.response_messages[0].text.text[0]



# FUNCTION FOR RETRY MECHANISM
def retry_dialogflow_response(chat_wrapper, user_message, attempts=3):
    no_answer_phrases = [
        "I'm sorry", "I apologize", "I can't answer this question",
        "I don't have that information", "Could you please rephrase your query?",
        "Unfortunately", "Sorry", "I am unable to provide instructions "
    ]
    
    for attempt in range(attempts):
        print(f"Attempt {attempt + 1} for query: '{user_message}'")  # Log the attempt number
        
        bot_response = chat_wrapper.detect_intent(user_message)
        print(f"Response from Dialogflow: '{bot_response}'")  # Log the response
        
        if not any(phrase.lower() in bot_response.lower() for phrase in no_answer_phrases):
            print("Received valid response, stopping retries.")
            return bot_response
        
        print("No valid response, retrying...")  # Log that retry is happening

    print(f"Max attempts reached. Returning last response: '{bot_response}'")
    modified_response = bot_response+" \nFor further reference and support you can click on Open Pdf Button or copy paste the given link to open the relevant page. Thank you!"
    return modified_response



# FUNCTION TO EXTRACT TEXT FROM PDF
def extract_text_from_pdf(pdf_path):
    text_data = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = clean_text(page.get_text("text"))
        text_data.append({"page_num": page_num, "text": text})
    return text_data



# FUNCTION TO CREATE GEMINI PRO API AND QUERY GEMINI
def gemini_pro_api(query):
    chat_wrapper = ChatWrapper()
    bot_response = retry_dialogflow_response(chat_wrapper, query)
    if '$request.knowledge.answer[0]' in bot_response or '$request.knowledge.answer[1]' in bot_response:
        return 'Could you please rephrase your query?'
    return bot_response



# FUNCTION TO FIND MOST RELEVANT PAGE
def find_most_relevant_page(relevant_text, text_data):
    # Use the original text without preprocessing
    texts = [entry['text'] for entry in text_data]
    relevant_text_vector = [relevant_text]

    # Create and apply TF-IDF Vectorizer
    vectorizer = create_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    relevant_text_vector = vectorizer.transform(relevant_text_vector)
    
    # Compute cosine similarities
    similarities = cosine_similarity(relevant_text_vector, tfidf_matrix).flatten()
    sorted_similarities = sorted([(sim, entry['page_num']) for sim, entry in zip(similarities, text_data) if sim > 0], key=lambda x: x[0], reverse=True)
   
    if not sorted_similarities:
        return None, []
    most_relevant_page = sorted_similarities[0][1]
    return most_relevant_page, sorted_similarities



# FUNCTION FOR QUERYING GEMINI
def query_pdf_gemini(query, text_data):
    relevant_text = gemini_pro_api(query)
    page_num, sorted_similarities = find_most_relevant_page(relevant_text, text_data)
    return relevant_text, page_num + 1, sorted_similarities if page_num is not None else []



# FUNCTION TO CREATE CHATBOT
def chatbot(query):
    return query_pdf_gemini(query, text_data)



# PDF PATH
pdf_path = r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\Demo_AI_EMR_Help_Docs_2.pdf"
text_data = extract_text_from_pdf(pdf_path)



#FUNCTION TO GENERATE PDF PAGE URL
def generate_pdf_page_url(pdf_path, page_num):
    pdf_path = pdf_path.replace("\\", "/")  # Use forward slashes
    encoded_path = urllib.parse.quote(pdf_path)
    return f"file:///{encoded_path}#page={page_num}"  # Properly formatted URL



# GRADIO UI
with gr.Blocks() as demo:
    gr.Markdown("## PDF Query Chatbot with Page Links")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
        suggestions_output = gr.Dropdown(label="Suggestions", choices=[], interactive=True)

    with gr.Row():
        text_output = gr.Textbox(label="Matched Text")
        page_number_output = gr.Number(label="Page Number", precision=0)
        page_link_output = gr.Button("Open PDF Page", visible=False)
        pdf_link_output = gr.Textbox(label="PDF Page Link", interactive=False)

    # FUNCTION TO SUGGEST USER QUERY 
    def suggest_query(user_input):
        suggestions = get_query_suggestions(user_input, stored_queries)
        return gr.update(choices=suggestions)

    # FUNCTION TO HANDLE QUERY SUBMISSION
    def submit_query(query):
        global stored_queries

        print(f"Received query: '{query}'")

        if query not in stored_queries:
            stored_queries.append(query)
            save_stored_queries(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\stored_queries.txt", stored_queries)
            print(f"Stored new query: '{query}'")

        relevant_text, page_num, _ = chatbot(query)
        print(f"Relevant text: '{relevant_text}'")
        print(f"Page number: {page_num}")

        # Function to generate the PDF page URL
        if page_num is not None:
            page_url = generate_pdf_page_url(pdf_path, page_num)
            return relevant_text, page_num, gr.update(visible=True), page_url  # Return the URL for the textbox
        else:
            return relevant_text, None, gr.update(visible=False), ""  # Set URL to empty if no page number

    query_input.change(suggest_query, query_input, suggestions_output)

    query_input.submit(submit_query, query_input, [text_output, page_number_output, page_link_output, pdf_link_output])
    
    suggestions_output.change(lambda selected: submit_query(selected), suggestions_output, [text_output, page_number_output, page_link_output, pdf_link_output])

    # FUNCTION TO OPEN PDF PAGE BUTTON
    def open_pdf_page(pdf_url):
        if pdf_url:
            webbrowser.get('chrome').open(pdf_url)  # Open the specific page in Chrome
            return "Opening PDF page in Chrome..."
        return "No valid page URL available."

    page_link_output.click(open_pdf_page, inputs=pdf_link_output, outputs=[])

# LAUNCHING GRADIO UI
demo.launch()
