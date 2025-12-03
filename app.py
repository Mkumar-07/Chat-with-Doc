import streamlit as st
from ingestion import PDFIngestor
from app_graph import PdfChat

st.set_page_config(page_title="Chat with Doc", page_icon="∂øç")

if "messages" not in st.session_state:
    st.session_state.messages =  [{"role": "assistant", "content": "Time to talk with your Doc!"}]
    st.session_state.app = None
with st.sidebar:
    st.info("This app uses the OpenAI API to generate text, please provide your API key."
            "\n If you don't have an API key, you can get one [here](https://platform.openai.com/signup)."
            "\n App keys are not stored or saved in any way.")

    chat_active = False
    st.divider()
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

def initialize_ingestor(pdf_files):
    retriever = PDFIngestor(pdfs=pdf_files).get_retriever()
    st.success("PDFs successfully uploaded")
    app = PdfChat(retriever).graph
    st.success("ChatBot successfully initialized")
    return app

with st.sidebar:
    if st.button("Initialize ChatBot", type="primary"):
        st.session_state.app = initialize_ingestor(pdf_files)

app = st.session_state.app
def generate_response(question):
    return app.invoke(input={"question": question})["response"]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:= st.chat_input(placeholder="Ask a question", disabled=chat_active):
    if not st.session_state.app:
        st.warning("Please upload a PDF and click 'Initialize ChatBot' first.")
    else:
        st.chat_message("user").markdown(question)

        st.session_state.messages.append({"role": "user", "content": question})

        response = generate_response(question)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
