import streamlit as st
import pandas as pd
import os
import tempfile
import time
import json
import logging
import gc
from typing import Dict, List, Any, Optional
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "Great Gray Analytics PA Report Chatbot"
MAX_FILE_SIZE_MB = 200
CACHE_DIR = ".streamlit/cache"

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if API key is in secrets
api_key_from_secrets = False
try:
    if "api_keys" in st.secrets:
        api_key = st.secrets["api_keys"]["anthropic"]
        os.environ["ANTHROPIC_API_KEY"] = api_key
        api_key_from_secrets = True
except Exception as e:
    pass  # Will handle via UI input

# Prompt template for PA Report specific context
PA_REPORT_PROMPT_TEMPLATE = """
You are an AI assistant that helps users analyze and understand Great Gray Analytics PA Report data.
The PA Report contains information about portfolios, assets under administration, performance metrics, and invoice details.

Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say you don't know.
Don't try to make up an answer. If the retrieved context doesn't provide enough information to answer confidently,
explain what additional data might be helpful.

When discussing financial metrics:
- Always specify the currency when mentioning monetary values
- Be precise about performance percentages
- Clarify time periods for any metrics
- Format large numbers with appropriate separators (e.g., $1,234,567.89)

Retrieved context:
{context}

Question: {question}

Answer:
"""

# Helper functions for caching
def get_cache_path(file_path: str, suffix: str = "vectorstore") -> str:
    """Generate a cache path based on file path and a suffix."""
    import hashlib
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}_{suffix}.pkl")

def save_to_cache(data: Any, cache_path: str) -> bool:
    """Save data to cache file."""
    try:
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}")
        return False

def load_from_cache(cache_path: str) -> Any:
    """Load data from cache file if it exists."""
    try:
        import pickle
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading from cache: {str(e)}")
    return None

def extract_sheet_data(file_path: str, max_sheets: int = 5, max_rows_per_sheet: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Extract data from Excel sheets in a memory-efficient way.
    
    Args:
        file_path: Path to Excel file
        max_sheets: Maximum number of sheets to process
        max_rows_per_sheet: Maximum rows to process per sheet
        
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    sheet_data = {}
    
    try:
        # Get sheet names
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names[:max_sheets]  # Limit number of sheets
        
        for sheet_name in sheet_names:
            try:
                # Try to get row count (this may fail for very large sheets)
                try:
                    sheet = xls.book.sheet_by_name(sheet_name)
                    row_count = getattr(sheet, 'nrows', 0)
                except:
                    # If we can't get exact count, read a single row to check if sheet has data
                    sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=1)
                    row_count = 1 if not sample.empty else 0
                
                if row_count > 0:
                    # If sheet is large, sample it
                    if row_count > max_rows_per_sheet:
                        # Read first rows
                        first_half = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows_per_sheet//2)
                        
                        try:
                            # Try to read last rows (may fail for extremely large sheets)
                            skip_rows = max(0, row_count - max_rows_per_sheet//2)
                            last_half = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows)
                            
                            # Combine
                            sheet_data[sheet_name] = pd.concat([first_half, last_half])
                        except:
                            # If reading the end fails, just use the first half
                            sheet_data[sheet_name] = first_half
                    else:
                        # Read entire sheet
                        sheet_data[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
            
            except Exception as e:
                logger.warning(f"Error reading sheet {sheet_name}: {str(e)}")
                continue
        
        return sheet_data
        
    except Exception as e:
        logger.error(f"Error extracting sheet data: {str(e)}")
        return {}

def process_dataframe_to_documents(df: pd.DataFrame, sheet_name: str) -> List[Document]:
    """
    Convert a DataFrame to a list of Document objects for LangChain.
    
    Args:
        df: DataFrame to process
        sheet_name: Name of the sheet for metadata
        
    Returns:
        List of Document objects
    """
    documents = []
    
    try:
        # Clean data
        df = df.fillna("")
        
        # Sample the dataframe if it's large
        if len(df) > 100:
            # Take first and last rows
            sampled_df = pd.concat([df.head(50), df.tail(50)])
        else:
            sampled_df = df
        
        # Convert to string representation
        text_content = f"Sheet: {sheet_name}\n\n"
        text_content += f"Columns: {', '.join(str(col) for col in sampled_df.columns)}\n\n"
        text_content += "Sample data:\n"
        text_content += sampled_df.to_string(max_rows=50, max_cols=10)
        
        # Create document
        doc = Document(
            page_content=text_content,
            metadata={"source": "PA Report", "sheet": sheet_name, "type": "sample_data"}
        )
        
        documents.append(doc)
        
        # Also create row-level documents for detailed analysis
        # Limit to first 100 rows to avoid memory issues
        for i, row in df.head(100).iterrows():
            # Create a text representation of the row
            row_content = f"Sheet: {sheet_name}, Row: {i}\n"
            for col, val in row.items():
                row_content += f"{col}: {val}\n"
            
            # Create document
            row_doc = Document(
                page_content=row_content,
                metadata={"source": "PA Report", "sheet": sheet_name, "row": i, "type": "row_data"}
            )
            
            documents.append(row_doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error converting DataFrame to documents: {str(e)}")
        return []

def create_vectorstore(documents: List[Document], chunk_size: int, chunk_overlap: int) -> Any:
    """
    Create a vector store from documents using ChromaDB.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Chroma vector store
    """
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split documents
        split_docs = text_splitter.split_documents(documents)
        
        # Use HuggingFace embeddings
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create in-memory Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=split_docs, 
            embedding=embedding_function,
            collection_name="pa_report",
            persist_directory=None  # In-memory only
        )
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def create_conversation_chain(vectorstore: Any, model_name: str) -> Any:
    """
    Create a conversation chain for the chatbot.
    
    Args:
        vectorstore: FAISS vector store
        model_name: Anthropic model name
        
    Returns:
        ConversationalRetrievalChain
    """
    try:
        # Create Claude LLM
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.1,
            streaming=True
        )
        
        # Create prompt
        prompt = PromptTemplate(
            template=PA_REPORT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return conversation_chain
        
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        return None

def display_welcome_screen():
    """Display the welcome screen."""
    st.markdown("""
    # Welcome to the Great Gray Analytics PA Report Chatbot
    
    This tool allows you to chat with your PA Report data through natural language.
    
    ## Getting Started
    
    1. Upload your PA Report Excel file using the sidebar
    2. Configure the AI model and processing options
    3. Click "Process File" to analyze the data
    4. Start chatting with your data!
    
    ## Example Questions
    
    - What is the total Assets Under Administration (AUA)?
    - How many portfolios are in the report?
    - What is the asset allocation across the portfolios?
    - Show me the performance summary
    - What are the invoice totals?
    
    ## Memory-Efficient Processing
    
    This application is designed to handle large Excel files by:
    
    - Processing the file in chunks
    - Sampling data from large sheets
    - Using vectorized search for efficient retrieval
    """)

def display_chat_interface():
    """Display the chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PA Report"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Generate response
                if "conversation_chain" in st.session_state:
                    with st.spinner("Thinking..."):
                        # Make API call
                        response = st.session_state.conversation_chain({"question": prompt})
                        
                        # Display response
                        message_placeholder.markdown(response["answer"])
                        
                        # Display sources
                        if "source_documents" in response and response["source_documents"]:
                            with st.expander("Sources"):
                                sources_seen = set()
                                
                                for i, doc in enumerate(response["source_documents"][:5]):
                                    source_key = f"{doc.metadata.get('sheet', 'Unknown')}-{doc.metadata.get('type', 'Unknown')}"
                                    
                                    if source_key not in sources_seen:
                                        sources_seen.add(source_key)
                                        st.markdown(f"**Source {i+1}:** Sheet '{doc.metadata.get('sheet', 'Unknown')}' - {doc.metadata.get('type', 'Unknown')}")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                else:
                    error_message = "Sorry, the chatbot is not properly initialized. Please reload the page and try again."
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

def main():
    """Main application function."""
    # Set page title and configuration
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    
    # Main title
    st.title("Great Gray Analytics PA Report Chatbot")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Claude configuration
        if not api_key_from_secrets:
            api_key = st.text_input("Anthropic API Key", type="password")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            st.success("Anthropic API key loaded from secrets!")
            
        model_name = st.selectbox("Claude Model", [
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307"
        ], index=0)
        
        # File upload
        file_section = st.container()
        with file_section:
            st.subheader("Upload PA Report")
            uploaded_file = st.file_uploader("Select Excel file", type=["xlsx", "xls"])
            
            if uploaded_file is not None:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"File size: {file_size_mb:.2f} MB")
                
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large! Maximum size is {MAX_FILE_SIZE_MB} MB")
                    uploaded_file = None
        
        # Processing options
        if uploaded_file is not None:
            options_section = st.container()
            with options_section:
                st.subheader("Processing Options")
                
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 
                                      help="Size of text chunks for processing")
                
                chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100,
                                         help="Overlap between chunks")
                
                max_sheets = st.number_input("Max Sheets to Process", 1, 20, 5,
                                           help="Maximum number of sheets to process")
                
                max_rows = st.number_input("Max Rows per Sheet", 500, 10000, 1000,
                                         help="Maximum rows to process per sheet")
                
                process_button = st.button("Process File", use_container_width=True)
                
                if process_button:
                    if not api_key_from_secrets and "ANTHROPIC_API_KEY" not in os.environ:
                        st.error("Please enter your Anthropic API key to proceed.")
                    else:
                        process_file(uploaded_file, model_name, chunk_size, chunk_overlap, max_sheets, max_rows)
    
    # Main content area
    if st.session_state.file_processed:
        display_chat_interface()
    else:
        display_welcome_screen()
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Great Gray Analytics")

def process_file(uploaded_file, model_name, chunk_size, chunk_overlap, max_sheets, max_rows):
    """Process the uploaded file."""
    st.session_state.file_processed = False
    
    with st.status("Processing PA Report file...", expanded=True) as status:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract data from sheets
            status.update(label="Reading Excel sheets...", state="running")
            sheet_data = extract_sheet_data(tmp_file_path, max_sheets, max_rows)
            
            if not sheet_data:
                status.update(label="No data found in the file.", state="error")
                return
            
            status.update(label=f"Found {len(sheet_data)} sheets.", state="running")
            
            # Convert to documents
            status.update(label="Converting data to documents...", state="running")
            documents = []
            
            for sheet_name, df in sheet_data.items():
                sheet_docs = process_dataframe_to_documents(df, sheet_name)
                documents.extend(sheet_docs)
                status.update(label=f"Processed sheet: {sheet_name}", state="running")
            
            # Create vector store
            status.update(label="Creating vector store...", state="running")
            vectorstore = create_vectorstore(documents, chunk_size, chunk_overlap)
            
            if vectorstore is None:
                status.update(label="Failed to create vector store.", state="error")
                return
            
            # Create conversation chain
            status.update(label="Setting up Claude model...", state="running")
            conversation_chain = create_conversation_chain(vectorstore, model_name)
            
            if conversation_chain is None:
                status.update(label="Failed to create conversation chain.", state="error")
                return
            
            # Save to session state
            st.session_state.conversation_chain = conversation_chain
            st.session_state.file_processed = True
            
            # Complete processing
            status.update(label="Processing complete! You can now chat with your PA Report.", state="complete")
            
        except Exception as e:
            status.update(label=f"Error processing file: {str(e)}", state="error")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            # Force garbage collection
            gc.collect()

if __name__ == "__main__":
    main()
