from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided PDF document.
Use the following context to answer the question. If you don't know the answer from the context, say so clearly.
Always be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, openai_api_key: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.qa_chain = None

        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, pdf_path: str) -> int:
        """Load PDF, split into chunks, embed and store. Returns number of chunks."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        chunks = self.text_splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return len(chunks)

    def query(self, question: str) -> tuple[str, list[str]]:
        """Query the RAG pipeline. Returns (answer, list of source chunk texts)."""
        if not self.qa_chain:
            raise ValueError("No PDF loaded. Call load_pdf() first.")

        result = self.qa_chain.invoke({"query": question})

        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return answer, sources

    def save_vectorstore(self, path: str):
        """Save FAISS index to disk for reuse."""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str):
        """Load a previously saved FAISS index."""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
