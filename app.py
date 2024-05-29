import google.cloud.bigquery as BigQuery
from google.oauth2 import service_account
import json
from langchain_google_vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import BigQueryLoader
from langchain.schema import format_document
import streamlit as st
import random

#Get global variables
with open("variables/global.json", "r") as variables:
    VARIABLES = json.load(variables)

with open("variables/examples.txt", "r") as examples:
    EXAMPLES = examples.read()

with open("variables/assistant_response.json", "r") as ass_response:
    ASSISTANT_RESPONSE = json.load(ass_response)

with open("variables/prompt_template.txt", "r") as prompt_template:
    PROMPT_TEMPLATE = prompt_template.read()

PROJECT_ID = VARIABLES["global"]["project_id"]
DATASET_ID = VARIABLES["global"]["dataset_id"]
LOCATION_ID = VARIABLES["global"]["location_id"]
VERTEX_AI_MODEL = VARIABLES["global"]["vertex_ai_model"]
CORA_RESPONSES = ASSISTANT_RESPONSE["cora_responses"]
CORA_NO_RESPONSES = ASSISTANT_RESPONSE["cora_no_responses"]

#Authentication Process
CREDENTIALS = service_account.Credentials.from_service_account_file(VARIABLES["global"]["service_account_key"])
CREDENTIALS = CREDENTIALS.with_scopes([scope.strip() for scope in VARIABLES["global"]["authentication_scope"].split(',')])

#Init Clients
BIGQUERY_CLIENT = BigQuery.Client(credentials=CREDENTIALS)

# LLMChain Inicialization
LLM = VertexAI(project=PROJECT_ID, location=LOCATION_ID, credentials=CREDENTIALS, model_name=VERTEX_AI_MODEL, max_output_tokens=8192, temperature=0)

#Functions
def generate_and_display_sql(user_question, examples, prompt_template):
    SCHEMAS_QUERY = f"""
    SELECT table_catalog, table_schema, table_name,	column_name, data_type
    FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`;
    """
    BQLOADER = BigQueryLoader(SCHEMAS_QUERY, page_content_columns=["table_catalog", "table_schema", "table_name",	"column_name", "data_type"], credentials=CREDENTIALS)
    SCHEMAS_DOCS = BQLOADER.load()
     
    chain = (
    {
    "schemas_data": lambda docs: "\n\n".join(
        format_document(doc, PromptTemplate.from_template("{page_content}"))
        for doc in docs
    ),
    }
    | PromptTemplate.from_template("""
                                   
            Prompt:
            """+prompt_template+"""

            Table Schema:
            {schemas_data}
                                        
            Question/SQL Generated Examples:
            """+examples+"""
                                        
            Question:
            """+user_question+"""
            """
    )
    | LLM

)
    # Process and Display Output
    result = chain.invoke(SCHEMAS_DOCS)
    clean_query = result.replace("```sql", "").replace("```", "") 
    try:
        df = BIGQUERY_CLIENT.query(clean_query).result().to_dataframe()
        sql_error = ""
    except Exception as e:
        df = []
        sql_error = e
    return (sql_error, df, clean_query)


#Build Frontend
st.set_page_config(layout="wide", page_title="GenAI Cortex Demo", page_icon="./images/CorAv2Streamlit.png")
with open( "css/style.css" ) as css:
    st.markdown(f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    st.image('./images/Coraheader970x250pxWhite.png')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=('./images/Userv2_128px.png' if message["role"] == 'human' else './images/CorAv2Streamlit.png')):
        st.markdown(message["content"])

if prompt := st.chat_input("Let me show my magic! Tell me what do you want!"):
    st.chat_message("human", avatar='./images/Userv2_128px.png').markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})
    
    with st.chat_message("assistant", avatar='./images/CorAv2Streamlit.png'):
         with st.spinner("Thinking..."):
            sql_error_result, result_df, clean_query = generate_and_display_sql(prompt, EXAMPLES, PROMPT_TEMPLATE)
            #st.markdown(result_query)
            if not sql_error_result or result_df:
                cora_generated_response = random.choice(CORA_RESPONSES)
                st.write(cora_generated_response)
                st.dataframe(result_df,use_container_width=True,hide_index=True) 
                with st.expander("See the related SQL Query"):
                    st.code(clean_query, language="sql", line_numbers=True)
                st.session_state.messages.append({"role": "assistant", "content": cora_generated_response})
            else:
                cora_generated_no_response = random.choice(CORA_NO_RESPONSES)
                st.write(cora_generated_no_response)
                with st.expander("See the related SQL Query"):
                    st.code(clean_query, language="sql", line_numbers=True)
                st.session_state.messages.append({"role": "assistant", "content": cora_generated_no_response})