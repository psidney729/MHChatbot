# Solution 1 - Normal Approach using Semantic Search


import psycopg2
from ..config import settings
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

conn = psycopg2.connect(f"host={settings.POSTGRES_HOST} dbname={settings.POSTGRES_DB} user={settings.POSTGRES_USER} password={settings.POSTGRES_PASSWORD}")
cur = conn.cursor()


async def aget_embedding_one(text, client):
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Define llm
llm = ChatOpenAI(model="gpt-4o")
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def semanticsearch(query):
    query_vector = await aget_embedding_one(query, client)
    cur.execute(f"""
        SELECT id, context, response, semantic_vector <=> '{query_vector}' as distance
        FROM conversationdb
        ORDER BY semantic_vector <=> '{query_vector}'
    """)
    
    # vectors = cur.fetchall()
    vectors = cur.fetchmany(5)

    return vectors
