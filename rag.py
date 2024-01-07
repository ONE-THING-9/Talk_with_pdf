from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import PyPDF2
import fitz


class RAG():
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    def parse_file(self,file_name):
        all_text = ""
        
        '''   
        loader = PyPDF2.PdfReader(file_name )
        
        for page_num in range(len(loader.pages)):
            page = loader.pages[page_num]
            all_text += page.extract_text()
        '''

        doc = fitz.open(stream=file_name.read(), filetype="pdf")
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()


        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([all_text])
        self.db = Chroma.from_documents(texts, self.embeddings)

    def retrieve(self,query):
        similarity  = self.db.similarity_search(query,3)
        context = ""
        print(similarity)
        for i in similarity:
            context += "\n\n" + i.page_content

        return context

    