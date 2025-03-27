import streamlit as st
import pandas as pd
import os
import tempfile
import time
import logging
from typing import Dict, List, Any
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "Great Gray Analytics PA Report Chatbot (Simple Version)"
MAX_FILE_SIZE_MB = 200
MAX_SHEETS = 5
MAX_ROWS_PER_SHEET = 1000

# Check if API key is in secrets
api_key_from_secrets = False
try:
    if "api_keys" in st.secrets:
        api_key = st.secrets["api_keys"]["anthropic"]
        os.environ["ANTHROPIC_API_KEY"] = api_key
        api_key_from_secrets = True
except Exception as e:
    pass  # Will handle via UI input

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
                        sheet_data[sheet_name] = pd.read_excel(
                            file_path, 
                            sheet_name=sheet_name, 
                            nrows=max_rows_per_sheet
                        )
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

def generate_sheet_summaries(sheet_data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Generate text summaries of each sheet.
    
    Args:
        sheet_data: Dictionary mapping sheet names to DataFrames
    
    Returns:
        List of text summaries
    """
    summaries = []
    
    for sheet_name, df in sheet_data.items():
        try:
            # Create basic summary
            summary = f"Sheet: {sheet_name}\n"
            summary += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
            summary += f"Column names: {', '.join(str(col) for col in df.columns)}\n\n"
            
            # Add sample data (first few rows)
            summary += "Sample data (first 5 rows):\n"
            summary += df.head(5).to_string() + "\n\n"
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary += "Summary statistics for numeric columns:\n"
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    try:
                        summary += f"{col} - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}\n"
                    except:
                        pass
            
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"Error summarizing sheet {sheet_name}: {str(e)}")
    
    return summaries

def create_conversation_chain(sheet_summaries: List[str], model_name: str) -> Any:
    """
    Create a simple conversation chain that doesn't use vector storage.
    
    Args:
        sheet_summaries: List of text summaries of each sheet
        model_name: Anthropic model name
    
    Returns:
        ConversationChain
    """
    try:
        # Create Claude LLM
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.1,
            streaming=True
        )
        
        # Combine summaries into a single context string
        context = "PA Report Data Summary:\n\n" + "\n".join(sheet_summaries)
        
        # Truncate if too long
        if len(context) > 25000:
            context = context[:25000] + "...(truncated)"
        
        # Create a system prompt with the context
        system_prompt = f"""You are an AI assistant that helps users analyze and understand Great Gray Analytics PA Report data.
        Use the following information about the Excel file to answer questions:
        
        {context}
        
        When discussing financial metrics:
        - Always specify the currency when mentioning monetary values
        - Be precise about performance percentages
        - Clarify time periods for any metrics
        - Format large numbers with appropriate separators (e.g., $1,234,567.89)
        
        If you're not sure about specific details, be honest about your limitations."""
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        
        # Create a wrapper to make it look like the retrieval chain interface
        class SimpleConversationWrapper:
            def __init__(self, conversation_chain):
                self.conversation = conversation_chain
                
            def __call__(self, inputs):
                question = inputs.get("question", "")
                response = self.conversation.predict(input=question)
                return {
                    "answer": response,
                    "source_documents": []  # No sources in simple mode
                }
        
        return SimpleConversationWrapper(conversation)
        
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def process_file(uploaded_file, model_name):
    """Process the uploaded file."""
    # Reset file processed flag at the beginning
    st.session_state.file_processed = False
    
    with st.status("Processing PA Report file...", expanded=True) as status:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract data from sheets
            status.update(label="Reading Excel sheets...", state="running")
            sheet_data = extract_sheet_data(tmp_file_path, MAX_SHEETS, MAX_ROWS_PER_SHEET)
            
            if not sheet_data:
                status.update(label="No data found in the file.", state="error")
                return
            
            status.update(label=f"Found {len(sheet_data)} sheets.", state="running")
            
            # Generate summaries
            status.update(label="Creating text summaries of the data...", state="running")
            sheet_summaries = generate_sheet_summaries(sheet_data)
            
            # Create conversation chain
            status.update(label="Setting up Claude model...", state="running")
            conversation_chain = create_conversation_chain(sheet_summaries, model_name)
            
            if conversation_chain is None:
                status.update(label="Failed to create conversation chain.", state="error")
                return
            
            # Save to session state
            st.session_state.conversation_chain = conversation_chain
            st.session_state.sheet_data = sheet_data
            
            # Set flag to indicate file is processed
            st.session_state.file_processed = True
            
            # Complete processing
            status.update(label="Processing complete! You can now chat with your PA Report.", state="complete")
            
            # Force a rerun to refresh the UI and show the chat interface
            st.experimental_rerun()
            
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

def display_welcome_screen():
    """Display the welcome screen."""
    st.markdown("""
    # Great Gray Analytics PA Report Chatbot (Simple Version)
    
    This simplified version of the chatbot allows you to chat with your PA Report data without using vector databases or complex embedding models. This makes it more compatible with different environments and easier to deploy.
    
    ## Getting Started
    
    1. Upload your PA Report Excel file
    2. Configure the Claude model
    3. Click "Process File"
    4. Start chatting with your data!
    
    ## Example Questions
    
    - What is the total Assets Under Administration?
    - How many portfolios are in the report?
    - What is the asset allocation across the portfolios?
    - Show me the performance summary
    """)

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
    st.title("Great Gray Analytics PA Report Chatbot (Simple Version)")
    
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
                else:
                    process_button = st.button("Process File", use_container_width=True)
                    
                    if process_button:
                        if not api_key_from_secrets and "ANTHROPIC_API_KEY" not in os.environ:
                            st.error("Please enter your Anthropic API key to proceed.")
                        else:
                            process_file(uploaded_file, model_name)
    
    # Main content area
    if st.session_state.file_processed:
        display_chat_interface()
    else:
        display_welcome_screen()
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Great Gray Analytics | Powered by Claude")

if __name__ == "__main__":
    main()
