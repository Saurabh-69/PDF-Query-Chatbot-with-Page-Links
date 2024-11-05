# THIS CODE FINDS THE MATCHED TEXT,THE FIRST IMAGE FROM THAT PAGE NUMEBR,
# RELEVANT PAGE NUMBER AND PDF LINK AND BUTTON TO OPEN THE PDF PAGE


import fitz  # PyMuPDF
import gradio as gr
from google.cloud import dialogflowcx_v3beta1 as dialogflowcx
from google.oauth2.service_account import Credentials
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
import re
import difflib
import urllib.parse
import webbrowser



# Specify the path to your Chrome executable
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))


# Text cleaning function
def clean_text(extracted_text):
    patterns_to_remove = [
        r"file:///C:/Users/pravin\.sharma/AppData/Local/Temp/~hh\d+\.htm \d{1,2}/\d{1,2}/\d{4}",
        r"Acknowledge Review"
    ]
    for pattern in patterns_to_remove:
        extracted_text = re.sub(pattern, '', extracted_text)
    return extracted_text

# Functions to read and write stored queries
def load_stored_queries(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def save_stored_queries(file_path, queries):
    with open(file_path, 'w') as file:
        for query in queries:
            file.write(query + '\n')

stored_queries = load_stored_queries(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\stored_queries.txt")

# Function to suggest queries based on user input
def get_query_suggestions(user_input, stored_queries):
    return difflib.get_close_matches(user_input, stored_queries, n=5, cutoff=0.3)

# Function to create TF-IDF vectorizer
def create_tfidf_vectorizer():
    return TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

# ChatWrapper class for Dialogflow
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

# Function to retry Dialogflow response
def retry_dialogflow_response(chat_wrapper, user_message, attempts=3):
    no_answer_phrases = [
        "I'm sorry", "I apologize", "I can't answer this question",
        "I don't have that information", "Could you please rephrase your query?",
        "Unfortunately"
    ]
    for attempt in range(attempts):
        bot_response = chat_wrapper.detect_intent(user_message)
        if not any(phrase.lower() in bot_response.lower() for phrase in no_answer_phrases):
            return bot_response
    return bot_response

# Function to extract text and images from PDF
def extract_text_and_images(pdf_path):
    text_data = []
    images = {}
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = clean_text(page.get_text("text"))
        text_data.append({"page_num": page_num, "text": text})
        image_list = page.get_images(full=True)
        page_images = []
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            image = Image.open(BytesIO(image_bytes))
            page_images.append(image)
        if page_images:
            images[page_num] = page_images
    return text_data, images

# Function to query Gemini
def gemini_pro_api(query):
    chat_wrapper = ChatWrapper()
    bot_response = retry_dialogflow_response(chat_wrapper, query)
    if '$request.knowledge.answer[0]' in bot_response or '$request.knowledge.answer[1]' in bot_response:
        return 'Could you please rephrase your query?'
    return bot_response

# Function to find the most relevant page
def find_most_relevant_page(relevant_text, text_data):
    texts = [entry['text'] for entry in text_data]
    relevant_text_vector = [relevant_text]
    vectorizer = create_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    relevant_text_vector = vectorizer.transform(relevant_text_vector)
    similarities = cosine_similarity(relevant_text_vector, tfidf_matrix).flatten()
    sorted_similarities = sorted([(sim, entry['page_num']) for sim, entry in zip(similarities, text_data) if sim > 0], key=lambda x: x[0], reverse=True)
    if not sorted_similarities:
        return None, []
    most_relevant_page = sorted_similarities[0][1]
    return most_relevant_page, sorted_similarities

# Function to query PDF using Gemini
def query_pdf_gemini(query, text_data, images):
    relevant_text = gemini_pro_api(query)
    page_num, sorted_similarities = find_most_relevant_page(relevant_text, text_data)
    if page_num is not None:
        page_images = images.get(page_num, [])
        image_list = [img for img in page_images]
        return relevant_text, image_list, page_num + 1, sorted_similarities
    return relevant_text, None, None, sorted_similarities

# Function to create chatbot
def chatbot(query):
    return query_pdf_gemini(query, text_data, images)

# PDF PATH
pdf_path = r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\Demo_AI_EMR_Help_Docs_2.pdf"
text_data, images = extract_text_and_images(pdf_path)


def generate_pdf_page_url(pdf_path, page_num):
    pdf_path = pdf_path.replace("\\", "/")
    encoded_path = urllib.parse.quote(pdf_path)
    return f"file:///{encoded_path}#page={page_num}"



# GRADIO UI
# GRADIO UI
# GRADIO UI
# GRADIO UI
with gr.Blocks() as demo:
    gr.Markdown("## PDF Query Chatbot with Related Screenshots and Page Links")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
        suggestions_output = gr.Dropdown(label="Suggestions", choices=[], interactive=True)

    with gr.Row():
        text_output = gr.Textbox(label="Matched Text")
        image_output = gr.Gallery(label="Related Screenshots", type="pil", show_label=True)

    with gr.Row():
        page_number_output = gr.Number(label="Page Number", precision=0)
        page_link_output = gr.Button("Open PDF Page", visible=True)
        pdf_link_output = gr.Textbox(label="PDF Page Link", interactive=False)

    def suggest_query(user_input):
        suggestions = get_query_suggestions(user_input, stored_queries)
        return gr.update(choices=suggestions)

    def submit_query(query):
        global stored_queries

        print(f"Received query: '{query}'")

        if query not in stored_queries:
            stored_queries.append(query)
            save_stored_queries(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\stored_queries.txt", stored_queries)
            print(f"Stored new query: '{query}'")

        relevant_text, related_images, page_num, _ = chatbot(query)
        print(f"Relevant text: '{relevant_text}'")
        print(f"Related images: {related_images}")
        print(f"Page number: {page_num}")

        # Generate the PDF page URL
        if page_num is not None:
            page_url = generate_pdf_page_url(pdf_path, page_num)
            return relevant_text, related_images, page_num, True, page_url  # Return the URL for the textbox
        else:
            return relevant_text, related_images, None, False, ""  # Set URL to empty if no page number

    query_input.change(suggest_query, query_input, suggestions_output)

    query_input.submit(submit_query, query_input, [text_output, image_output, page_number_output, page_link_output, pdf_link_output])
    
    suggestions_output.change(lambda selected: submit_query(selected), suggestions_output, [text_output, image_output, page_number_output, page_link_output, pdf_link_output])

    # Open PDF Page button
    def open_pdf_page(pdf_url):
        if pdf_url:
            webbrowser.get('chrome').open(pdf_url)
            return "Opening PDF page in Chrome..."
        return "No valid page URL available."

    page_link_output.click(open_pdf_page, inputs=pdf_link_output, outputs=[])

# LAUNCHING GRADIO UI
demo.launch()

