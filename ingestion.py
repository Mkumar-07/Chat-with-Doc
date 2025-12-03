from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import tempfile

class PDFIngestor:
    def __init__(self, pdfs):
        self.pdfs = pdfs
        self.doc_list = self.get_docs()

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=0
        )

        self.doc_splits = self.text_splitter.split_documents(self.doc_list)

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = FAISS.from_documents(
            documents=self.doc_splits,
            docstore=InMemoryDocstore(),
            embedding=self.embeddings
        )

    def get_docs(self):
        doc_list = []
        for pdf_file in self.pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_file.getvalue())
                temp_pdf.flush()
                doc_list.append(PyPDFLoader(temp_pdf.name).load())
        doc_list = [item for sublist in doc_list for item in sublist]
        return doc_list

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
