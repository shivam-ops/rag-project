from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from .config import Config


class RAGSystem:
    def __init__(self, file_path: str = Config.DATA_PATH):
        self.file_path = file_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name=Config.MODEL_NAME,
            temperature=0.7
        )
        self.vector_store = None
        self.chat_history = []

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

            # Ensure chunks are proper Document objects
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
            # Get relevant documents
            docs = self.vector_store.similarity_search(question, k=Config.TOP_K_RESULTS)

            # Prepare context
            context = "\n\n".join(doc.page_content for doc in docs)

            # Create messages using the proper message schema
            messages = [
                SystemMessage(content="You are a helpful assistant. Use the following context to answer the question."),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
            ]

            # Get response from OpenAI
            response = self.llm.invoke(messages)

            # Extract the response content
            response_content = response.content if hasattr(response, 'content') else str(response)

            # Update chat history
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