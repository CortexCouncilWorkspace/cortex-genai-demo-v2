import google.cloud.bigquery as bq
import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import BigQueryLoader
from langchain.schema import format_document
import random
from google.oauth2 import service_account

project_id = "ce-sap-latam-demo"
dataset_id = "cortex_sap_cora"
location_id = "us-central1"
vertex_ai_model_name = "gemini-1.0-pro" 
scope = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/bigquery']
credentials = service_account.Credentials.from_service_account_file('config/ce-sap-latam-demo-8e943944d80b.json')
credentials = credentials.with_scopes(scope) 
client = bq.Client(credentials=credentials)

similar_questions = f'''
Question:
liste os materiais com prima na descricao, traga tambem a descricao do material
Answer:
SELECT
    MaterialsMD.MaterialNumber_MATNR,
    MaterialsMD.MaterialText_MAKTX
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS MaterialsMD
WHERE
    UPPER(MaterialsMD.MaterialText_MAKTX) LIKE '%PRIMA%'

Question:
quais os materiais marcados para eliminacao, traga a descricao do material tambem

Answer:
SELECT
    MaterialsMD.MaterialNumber_MATNR,
    MaterialsMD.MaterialText_MAKTX
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS MaterialsMD
WHERE
    UPPER(MaterialsMD.FlagMaterialforDeletion_LVORM) LIKE '%X%'

Question:
liste os materiais marcados para eliminacao que tem estoque maior que 0, trazer tambem a descricao do material e quantidade

Answer:
SELECT
    MaterialsMD.MaterialNumber_MATNR,
    MaterialsMD.MaterialText_MAKTX,
    Stock_NonValuated.ValuatedUnrestrictedUseStock_LABST
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS MaterialsMD
INNER JOIN
    `ce-sap-latam-demo.cortex_sap_cora.Stock_NonValuated` AS Stock_NonValuated
ON MaterialsMD.MaterialNumber_MATNR = Stock_NonValuated.MaterialNumber_MATNR
WHERE
    MaterialsMD.FlagMaterialforDeletion_LVORM = 'X'
    AND Stock_NonValuated.ValuatedUnrestrictedUseStock_LABST > 0;


Question:
liste os materiais tem estoque maior que 50, trazer tambem a descricao do material, quantidade e a planta

Answer:
SELECT
    t1.MaterialNumber_MATNR,
    t1.MaterialText_MAKTX,
    t2.ValuatedUnrestrictedUseStock_LABST,
    t2.Plant_WERKS
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS t1
INNER JOIN
    `ce-sap-latam-demo.cortex_sap_cora.Stock_NonValuated` AS t2
ON t1.MaterialNumber_MATNR = t2.MaterialNumber_MATNR
WHERE t2.ValuatedUnrestrictedUseStock_LABST > 50


Question:
liste os materiais com codigo NCM "NCM02", traga tambem a descricao

Answer:
SELECT
    MaterialsMD.MaterialNumber_MATNR,
    MaterialsMD.MaterialText_MAKTX
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS MaterialsMD
WHERE
    UPPER(MaterialsMD.ControlCodeTaxesNCMCode_STEUC) LIKE '%NCM02%'

Question:
quero a descrição, quantidade, deposito, centro do material RAW02

Answer:
SELECT
    t1.MaterialDescriptionForMatchcodes_MAKTG,
    t2.ValuatedUnrestrictedUseStock_LABST,
    t2.StorageLocation_LGORT,
    t2.Plant_WERKS
FROM
    `ce-sap-latam-demo.cortex_sap_cora.MaterialsMD` AS t1
INNER JOIN
    `ce-sap-latam-demo.cortex_sap_cora.Stock_NonValuated` AS t2 ON t1.MaterialNumber_MATNR = t2.MaterialNumber_MATNR
WHERE
    UPPER(t1.MaterialNumber_MATNR) LIKE '%RAW02%'
'''

def generate_and_display_sql(user_question):
    # BigQuery Loader
    query = f"""
    SELECT table_catalog, table_schema, table_name,	column_name, data_type
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`;
    """
    loader = BigQueryLoader(query, page_content_columns=["table_catalog", "table_schema", "table_name",	"column_name", "data_type"], credentials=credentials)
    data = loader.load()

    # LLMChain Setup
    llm = VertexAI(project=project_id, location=location_id, credentials=credentials, model_name=vertex_ai_model_name, max_output_tokens=8192, temperature=0)
    chain = (
    {
        "content": lambda docs: "\n\n".join(
            format_document(doc, PromptTemplate.from_template("{page_content}"))
            for doc in docs
        )
    }
    | PromptTemplate.from_template("""
            You are a BigQuery SQL guru. Write a SQL comformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

            **Instructions:**

            * **Essential:**
                * Omit column aliases.
                * Capitalize all columns, proper names, and search strings values in the SQL WHERE clause by using the BigQuery 'UPPER' function (e.g., `WHERE UPPER(column_name) LIKE '%SEARCH_STRING%'`).
                * Use single quotes for string literals (e.g., `WHERE column_name LIKE '%value%'`).
                * Employ `LIKE` (not `=`) for string column comparisons.
                * Do not make any explanation, the output must be only SQL query.
                * Always include underscores: When referencing column names, always include underscores as part of the name (e.g., MaterialNumber_MATNR).
                * Return syntactically and symantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
            * **Date Handling:**
                * Adhere to the YYYY-MM-DD format for dates.
                * Employ the `BETWEEN` function for date comparisons.
                * Adapting to Partial Dates: If the user supplies only:
                    * Year: Construct the date range as 'YYYY-01-01' AND 'YYYY-12-31'.
                    * Year and Month: Generate the range 'YYYY-MM-01' and the last day of that month using BigQuery date functions.
                * Example: If the user provides '2023', the range becomes '2023-01-01' AND '2023-12-31'.
            * **Real-world Awareness:**
                * Do not invent data. Rely solely on information present in the BigQuery table schemas.
                * Do not make assumptions or extrapolations beyond the explicitly provided information in the BigQuery table schemas.
                * If a question cannot be answered with the available data, clearly state that the information is not available.
            * **GoogleSQL Optimization:**
                * Prioritize query efficiency. Consider factors like appropriate joins, filtering, and aggregation to minimize query costs and execution time.
                * Use `WHERE` clauses before `JOIN`s to filter data early.
                * Consider using `LIMIT` for large datasets to avoid unnecessary processing.
            * **Clarity and Readability:**
                * Format queries for easy understanding, using consistent indentation and spacing.

            Table Schema:
            {content}
                                        
            Question/SQL Generated Examples:
            """+similar_questions+"""
                                        
            Question:
            """+user_question+"""
            """
    )
    | llm

)
    # Process and Display Output
    result = chain.invoke(data)
    clean_query = result.replace("```sql", "")
    clean_query = clean_query.replace("```", "") 
    try:
        df = client.query(clean_query).result().to_dataframe()
        sql_error = ""
    except Exception as e:
        df = []
        sql_error = e
    return (sql_error, df, clean_query)

#CoraAI Responses:
cora_responses = [
                 "I'd be glad to help! Here's your answer!",
                 "Great question! Let me get your request...",
                 "Absolutely!",
                 "Of course! Here's the data requested."
                 ]
cora_no_responses = [
                "Hmm, I'm still learning about that. Could you rephrase your question, or provide more context?",
                "I'm not able to find a direct answer right now.",
                "That's a bit outside of my area of expertise.",
                "I'm having trouble to find this information.",
                "It seems like I might need some more training on that topic."
                 ]

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
            sql_error_result, result_df, clean_query = generate_and_display_sql(prompt)
            #st.markdown(result_query)
            if not sql_error_result or result_df:
                cora_generated_response = random.choice(cora_responses)
                st.write(cora_generated_response)
                st.dataframe(result_df,use_container_width=True,hide_index=True) 
                with st.expander("See the related SQL Query"):
                    st.code(clean_query, language="sql", line_numbers=True)
                st.session_state.messages.append({"role": "assistant", "content": cora_generated_response})
            else:
                cora_generated_no_response = random.choice(cora_no_responses)
                st.write(cora_generated_no_response)
                with st.expander("See the related SQL Query"):
                    st.code(clean_query, language="sql", line_numbers=True)
                st.session_state.messages.append({"role": "assistant", "content": cora_generated_no_response})