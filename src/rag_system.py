from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from .config import Config


class RAGSystem:
    def __init__(self, file_path: str = Config.DATA_PATH):
        self.file_path = file_path

        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=Config.AZURE_EMBEDDINGS_DEPLOYMENT,
            openai_api_version=Config.AZURE_API_VERSION,
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
        )

        # Initialize Azure ChatOpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=Config.AZURE_DEPLOYMENT,
            openai_api_version=Config.AZURE_API_VERSION,
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            temperature=0.7
        )

        self.vector_store = None
        self.chat_history = []

    # Rest of the code remains the same
    def load_and_process_document(self):
        print("Loading and processing document...")
        try:
            loader = TextLoader(self.file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Document split into {len(chunks)} chunks")

            processed_chunks = []
            for chunk in chunks:
                if isinstance(chunk, Document):
                    processed_chunks.append(chunk)
                else:
                    processed_chunks.append(Document(
                        page_content=chunk.page_content,
                        metadata=chunk.metadata if hasattr(chunk, 'metadata') else {}
                    ))

            self.vector_store = FAISS.from_documents(processed_chunks, self.embeddings)
            print("Document processing completed successfully")

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise

    def query(self, question: str):
        if not self.vector_store:
            raise ValueError("Please load and process the document first")

        try:
            docs = self.vector_store.similarity_search(question, k=Config.TOP_K_RESULTS)
            context = "\n\n".join(doc.page_content for doc in docs)

            messages = [
                SystemMessage(content="You are a helpful assistant. Use the following context to answer the question."),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
            ]

            response = self.llm.invoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)

            self.chat_history.append((question, response_content))

            return {
                "answer": response_content,
                "context": context
            }

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise

    def clear_chat_history(self):
        self.chat_history = []
        print("Chat history cleared")