import streamlit as st
import pandas as pd
import os
import tempfile
import anthropic

# Constants
APP_TITLE = "Great Gray Analytics PA Report Chatbot"
MAX_FILE_SIZE_MB = 200
MAX_SHEETS = 1
MAX_ROWS_PER_SHEET = 200000

# Check if API key is in secrets
api_key_from_secrets = False
try:
    if "api_keys" in st.secrets:
        api_key = st.secrets["api_keys"]["anthropic"]
        os.environ["ANTHROPIC_API_KEY"] = api_key
        api_key_from_secrets = True
except Exception as e:
    pass  # Will handle via UI input

def extract_sheet_data(file_path, max_sheets=1, max_rows_per_sheet=200000):
    """Extract data from Excel sheets."""
    sheet_data = {}
    
    try:
        # Get sheet names
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names[:max_sheets]  # Limit number of sheets
        
        for sheet_name in sheet_names:
            try:
                # Read with limit on rows
                sheet_data[sheet_name] = pd.read_excel(
                    file_path, 
                    sheet_name=sheet_name, 
                    nrows=max_rows_per_sheet
                )
            except Exception as e:
                st.warning(f"Error reading sheet {sheet_name}: {str(e)}")
        
        return sheet_data
        
    except Exception as e:
        st.error(f"Error extracting sheet data: {str(e)}")
        return {}

def generate_sheet_summaries(sheet_data):
    """Generate text summaries of each sheet."""
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

def process_file(uploaded_file, model_name):
    """Process the uploaded file."""
    # Reset file processed flag
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
            
            # Combine summaries
            combined_summary = "\n\n".join(sheet_summaries)
            
            # Truncate if too long (Claude has a context limit)
            if len(combined_summary) > 25000:
                combined_summary = combined_summary[:25000] + "...(truncated)"
            
            # Save to session state
            st.session_state.sheet_data = sheet_data
            st.session_state.sheet_summary = combined_summary
            
            # Reset messages and create system message
            st.session_state.messages = []
            
            # Create system message with context
            system_message = f"""You are an AI assistant that helps users analyze and understand Great Gray Analytics PA Report data.
            Use the following information about the Excel file to answer questions:
            
            {combined_summary}
            
            When discussing financial metrics:
            - Always specify the currency when mentioning monetary values
            - Be precise about performance percentages
            - Clarify time periods for any metrics
            - Format large numbers with appropriate separators (e.g., $1,234,567.89)
            
            If you're not sure about specific details, be honest about your limitations."""
            
            st.session_state.system_message = system_message
            
            # Initialize Anthropic client
            try:
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                st.session_state.client = client
                
                # Set flag to indicate file is processed
                st.session_state.file_processed = True
                st.session_state.model = model_name
                
                # Complete processing
                status.update(label="Processing complete! You can now chat with your PA Report.", state="complete")
                
                # Force a rerun to refresh the UI
                st.experimental_rerun()
                
            except Exception as e:
                status.update(label=f"Error initializing Anthropic client: {str(e)}", state="error")
                import traceback
                st.error(traceback.format_exc())
            
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
    # Initialize chat context in session state if not already there
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Ensure system message is initialized
    if "system_message" not in st.session_state:
        # Create system message with context
        system_message = f"""You are an AI assistant that helps users analyze and understand Great Gray Analytics PA Report data.
        Use the following information about the Excel file to answer questions:
        
        {st.session_state.sheet_summary if "sheet_summary" in st.session_state else "No data loaded yet."}
        
        When discussing financial metrics:
        - Always specify the currency when mentioning monetary values
        - Be precise about performance percentages
        - Clarify time periods for any metrics
        - Format large numbers with appropriate separators (e.g., $1,234,567.89)
        
        If you're not sure about specific details, be honest about your limitations."""
        
        st.session_state.system_message = system_message
    
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
                # Prepare messages for API call - system message is separate parameter
                messages = []
                
                # Add conversation history (system message is handled separately)
                for msg in st.session_state.messages:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Make API call to Claude
                with st.spinner("Thinking..."):
                    response = st.session_state.client.messages.create(
                        model=st.session_state.model,
                        system=st.session_state.system_message,  # System message as separate parameter
                        messages=messages,
                        max_tokens=1000
                    )
                    
                    # Display response
                    assistant_response = response.content[0].text
                    message_placeholder.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_message)
                import traceback
                st.error(traceback.format_exc())

def display_welcome_screen():
    """Display the welcome screen."""
    st.markdown("""
    # Great Gray Analytics PA Report Chatbot
    
    This application allows you to chat with your PA Report data using natural language.
    
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
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    
    if "sheet_summary" not in st.session_state:
        st.session_state.sheet_summary = "No data loaded yet."
    
    if "system_message" not in st.session_state:
        st.session_state.system_message = """You are an AI assistant that helps users analyze and understand Great Gray Analytics PA Report data.
        No data has been loaded yet, so you can only answer general questions."""
    
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
            
        model_options = {
            "Claude 3 Opus": "claude-3-opus-20240229",
            "Claude 3 Sonnet": "claude-3-sonnet-20240229",
            "Claude 3 Haiku": "claude-3-haiku-20240307"
        }
        
        model_selection = st.selectbox(
            "Claude Model",
            options=list(model_options.keys()),
            index=1  # Default to Sonnet
        )
        model_name = model_options[model_selection]
        
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
