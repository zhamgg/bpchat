import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from pathlib import Path
import json
import altair as alt
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional

# Set page configuration
st.set_page_config(
    page_title="BoardingPass PA Report Assistant",
    page_icon="📊",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .chat-message-user {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #E5E7EB;
    }
    .chat-message-bot {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #DBEAFE;
        border-left: 5px solid #3B82F6;
    }
    .dashboard-card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .stDataFrame {
        margin: 1rem 0;
    }
    .stSelectbox label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and process the data
@st.cache_data
def load_data(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Convert date columns to datetime
        date_columns = ['Request Date', 'Estimated Funding Date', 'Report As of Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill NaN values with appropriate placeholders
        df = df.fillna({
            'Request ID': 'Unknown',
            'Source': 'Unknown',
            'Submitter': 'Unknown',
            'Request Status': 'Unknown',
            'Status Detail': 'No details available',
            'TAAcctNum': 'N/A',
            'Advisor CRD Number': 'N/A',
            'Estimated Funding Amount': 0,
            'Fund Net Assets': 0,
            'Percentage of Fund': 0
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Natural Language Processing Utilities
class QueryProcessor:
    """Class to handle natural language query processing"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.keywords = self._build_keywords()
        
    def _build_keywords(self) -> Dict[str, List[str]]:
        """Build a dictionary of keywords and their synonyms"""
        return {
            'count': ['how many', 'count', 'total number', 'quantity'],
            'request': ['request', 'requests', 'pa', 'pas', 'participant agreement'],
            'plan': ['plan', 'plans', '401k', 'retirement plan'],
            'fund': ['fund', 'funds', 'investment', 'investments'],
            'advisor': ['advisor', 'advisors', 'financial advisor', 'fa'],
            'recordkeeper': ['recordkeeper', 'record keeper', 'record-keeper', 'rk'],
            'status': ['status', 'state', 'condition', 'stage'],
            'funding': ['funding', 'money', 'amount', 'dollar', 'dollars', '$'],
            'total': ['total', 'sum', 'aggregate', 'overall'],
            'average': ['average', 'avg', 'mean', 'typical'],
            'highest': ['highest', 'max', 'maximum', 'largest', 'biggest', 'most'],
            'lowest': ['lowest', 'min', 'minimum', 'smallest', 'least'],
            'information': ['information', 'details', 'about', 'tell me about', 'what do you know about'],
            'submitter': ['submitter', 'submitted by', 'who submitted', 'entered by'],
            'source': ['source', 'origin', 'where from', 'generated from'],
            'popular': ['popular', 'most common', 'frequent', 'top'],
            'help': ['help', 'what can you do', 'capabilities', 'functions', 'features']
        }
    
    def contains_keywords(self, query: str, keyword_category: str) -> bool:
        """Check if the query contains any keywords from the specified category"""
        if keyword_category not in self.keywords:
            return False
        
        return any(kw in query.lower() for kw in self.keywords[keyword_category])
    
    def extract_plan_name(self, query: str) -> Optional[str]:
        """Extract a plan name from the query if present"""
        for plan_name in self.df['Plan Name'].unique():
            if plan_name.lower() in query.lower():
                return plan_name
        return None
    
    def extract_fund_name(self, query: str) -> Optional[str]:
        """Extract a fund name from the query if present"""
        for fund_name in self.df['Fund Name'].unique():
            if fund_name.lower() in query.lower():
                return fund_name
        return None
    
    def extract_advisor_name(self, query: str) -> Optional[str]:
        """Extract an advisor name from the query if present"""
        for advisor_name in self.df['Advisor Name'].unique():
            if advisor_name.lower() in query.lower():
                return advisor_name
        return None
    
    def extract_date_range(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract date range from the query"""
        # This is a simplified implementation - in a real-world scenario, 
        # you might use a more sophisticated date parsing library
        start_date = None
        end_date = None
        
        # Simple regex to match dates like MM/DD/YY or MM/DD/YYYY
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})'
        dates = re.findall(date_pattern, query)
        
        if len(dates) >= 2:
            try:
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[1])
            except:
                pass
        
        return start_date, end_date
    
    def process_query(self, query: str) -> str:
        """Process the user query and generate a response"""
        try:
            # Basic statistics queries
            if self.contains_keywords(query, 'count'):
                if self.contains_keywords(query, 'request'):
                    return f"There are {self.df['Request ID'].nunique()} unique requests in the report."
                elif self.contains_keywords(query, 'plan'):
                    return f"There are {self.df['Plan Name'].nunique()} unique plans in the report."
                elif self.contains_keywords(query, 'fund'):
                    return f"There are {self.df['Fund Name'].nunique()} unique funds in the report."
                elif self.contains_keywords(query, 'advisor'):
                    return f"There are {self.df['Advisor Name'].nunique()} unique advisors in the report."
                elif self.contains_keywords(query, 'recordkeeper'):
                    return f"There are {self.df['Recordkeeper Name'].nunique()} unique recordkeepers in the report."
            
            # Status related queries
            elif self.contains_keywords(query, 'status'):
                status_counts = self.df['Request Status'].value_counts().to_dict()
                response = "Here's the breakdown of request statuses:\n\n"
                for status, count in status_counts.items():
                    response += f"- {status}: {count} requests\n"
                return response
            
            # Funding related queries
            elif self.contains_keywords(query, 'funding'):
                if self.contains_keywords(query, 'total'):
                    total_funding = self.df['Estimated Funding Amount'].sum()
                    return f"The total estimated funding amount is ${total_funding:,.2f}."
                elif self.contains_keywords(query, 'average'):
                    avg_funding = self.df['Estimated Funding Amount'].mean()
                    return f"The average estimated funding amount is ${avg_funding:,.2f}."
                elif self.contains_keywords(query, 'highest'):
                    max_funding = self.df['Estimated Funding Amount'].max()
                    max_plan = self.df[self.df['Estimated Funding Amount'] == max_funding]['Plan Name'].iloc[0]
                    return f"The highest estimated funding amount is ${max_funding:,.2f} for the {max_plan} plan."
                elif self.contains_keywords(query, 'lowest'):
                    min_funding = self.df[self.df['Estimated Funding Amount'] > 0]['Estimated Funding Amount'].min()
                    min_plan = self.df[self.df['Estimated Funding Amount'] == min_funding]['Plan Name'].iloc[0]
                    return f"The lowest positive estimated funding amount is ${min_funding:,.2f} for the {min_plan} plan."
                else:
                    total_funding = self.df['Estimated Funding Amount'].sum()
                    avg_funding = self.df['Estimated Funding Amount'].mean()
                    return f"The total estimated funding amount is ${total_funding:,.2f}, with an average of ${avg_funding:,.2f} per request."
            
            # Plan specific queries
            elif self.contains_keywords(query, 'plan') and self.contains_keywords(query, 'information'):
                plan_name = self.extract_plan_name(query)
                
                if plan_name:
                    plan_data = self.df[self.df['Plan Name'] == plan_name].iloc[0]
                    response = f"**Plan Details for {plan_name}**\n\n"
                    response += f"- Plan ID: {plan_data['PlanID']}\n"
                    response += f"- Plan Sponsor: {plan_data['Plan Sponsor Name']}\n"
                    response += f"- Plan Type: {plan_data['Plan Type']}\n"
                    response += f"- Recordkeeper: {plan_data['Recordkeeper Name']}\n"
                    response += f"- Advisor: {plan_data['Advisor Name']} ({plan_data['Advisor Firm Name']})\n"
                    
                    # Get fund information for this plan
                    plan_funds = self.df[self.df['Plan Name'] == plan_name]
                    if len(plan_funds) > 1:
                        response += f"\nThis plan has {len(plan_funds)} funds:\n"
                        for _, fund_row in plan_funds.iterrows():
                            fund_amount = fund_row['Estimated Funding Amount']
                            response += f"- {fund_row['Fund Name']}: ${fund_amount:,.2f}\n"
                    
                    return response
                else:
                    plan_list = self.df['Plan Name'].unique()[:5]  # Show first 5 plans
                    response = "Please specify which plan you'd like information about. Here are some plans in the report:\n\n"
                    for plan in plan_list:
                        response += f"- {plan}\n"
                    
                    if len(self.df['Plan Name'].unique()) > 5:
                        response += f"\n...and {len(self.df['Plan Name'].unique()) - 5} more plans."
                    
                    return response
            
            # Fund specific queries
            elif self.contains_keywords(query, 'fund') and self.contains_keywords(query, 'information'):
                fund_name = self.extract_fund_name(query)
                
                if fund_name:
                    fund_data = self.df[self.df['Fund Name'] == fund_name]
                    response = f"**Fund Details for {fund_name}**\n\n"
                    
                    # Get common details
                    first_row = fund_data.iloc[0]
                    response += f"- Fund ID: {first_row['Fund ID']}\n"
                    response += f"- CUSIP: {first_row['Cusip']}\n"
                    response += f"- Net Assets: ${first_row['Fund Net Assets']:,.2f}\n\n"
                    
                    # Get plans investing in this fund
                    response += f"This fund appears in {len(fund_data)} plans:\n"
                    for _, row in fund_data.iterrows():
                        fund_amount = row['Estimated Funding Amount']
                        response += f"- {row['Plan Name']}: ${fund_amount:,.2f}\n"
                    
                    return response
                else:
                    fund_list = self.df['Fund Name'].unique()[:5]  # Show first 5 funds
                    response = "Please specify which fund you'd like information about. Here are some funds in the report:\n\n"
                    for fund in fund_list:
                        response += f"- {fund}\n"
                    
                    if len(self.df['Fund Name'].unique()) > 5:
                        response += f"\n...and {len(self.df['Fund Name'].unique()) - 5} more funds."
                    
                    return response
            
            # Submitter related queries
            elif self.contains_keywords(query, 'submitter'):
                submitter_counts = self.df['Submitter'].value_counts().to_dict()
                response = "Here's the breakdown of requests by submitter:\n\n"
                for submitter, count in submitter_counts.items():
                    response += f"- {submitter}: {count} requests\n"
                return response
            
            # Most popular funds
            elif self.contains_keywords(query, 'fund') and self.contains_keywords(query, 'popular'):
                fund_counts = self.df['Fund Name'].value_counts()
                top_funds = fund_counts.head(5)
                response = "The most popular funds in the report are:\n\n"
                for fund, count in top_funds.items():
                    response += f"- {fund}: {count} occurrences\n"
                return response
            
            # Source related queries
            elif self.contains_keywords(query, 'source'):
                source_counts = self.df['Source'].value_counts().to_dict()
                response = "Here's the breakdown of requests by source:\n\n"
                for source, count in source_counts.items():
                    response += f"- {source}: {count} requests\n"
                return response
            
            # Help query
            elif self.contains_keywords(query, 'help'):
                return """
                I can help you analyze the BoardingPass PA Report data. Here are some things you can ask me:
                
                - How many requests/plans/funds are in the report?
                - What's the status breakdown of requests?
                - What's the total/average estimated funding amount?
                - What's the highest/lowest estimated funding amount?
                - Give me information about a specific plan (e.g., "Tell me about DALLAS CUSTOM ROOFING 401(K) PLAN")
                - Give me information about a specific fund
                - Who are the submitters in the report?
                - What are the most popular funds?
                - What are the sources of requests?
                - Show me plans with the highest funding amounts
                
                Feel free to ask any other questions about the data!
                """
            
            # Advanced analysis - plans with highest funding
            elif "highest funding" in query.lower() or "largest plans" in query.lower():
                plan_funding = self.df.groupby('Plan Name')['Estimated Funding Amount'].sum().sort_values(ascending=False)
                top_plans = plan_funding.head(5)
                
                response = "Plans with the highest total estimated funding amounts:\n\n"
                for plan, amount in top_plans.items():
                    response += f"- {plan}: ${amount:,.2f}\n"
                
                return response
            
            # Default response for unrecognized queries
            else:
                return "I'm not sure how to answer that question. Please try rephrasing or ask about specific data points in the PA Report. Type 'help' to see what I can do."
        
        except Exception as e:
            return f"I encountered an error trying to answer your question: {str(e)}"
