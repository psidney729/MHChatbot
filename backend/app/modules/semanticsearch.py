# Solution 1 - Normal Approach using Semantic Search


import psycopg2
from ..config import settings
from openai import OpenAI
from langchain_openai import ChatOpenAI

conn = psycopg2.connect(f"host={settings.POSTGRES_HOST} dbname={settings.POSTGRES_DB} user={settings.POSTGRES_USER} password={settings.POSTGRES_PASSWORD}")
cur = conn.cursor()

# Define llm
llm = ChatOpenAI(model="gpt-4o")
client = OpenAI()

def semanticsearch(query):
    query_vector = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    cur.execute(f"""
        SELECT id, context, response, semantic_vector <=> '{query_vector}' as distance
        FROM conversationdb
        ORDER BY semantic_vector <=> '{query_vector}'
    """)
    
    # vectors = cur.fetchall()
    vectors = cur.fetchmany(5)

    return vectors
