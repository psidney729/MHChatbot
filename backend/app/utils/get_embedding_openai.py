# Get OpenAI Embedding

import json
import asyncio
from openai import AsyncOpenAI
import datetime
from ..config.config import settings


async def aget_embedding_one(text, client):
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

async def aget_embedding_chunk(chunk):
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    tasks = [aget_embedding_one(text, client) for text in chunk]
    return await asyncio.gather(*tasks)

def get_embedding_openai(texts, chunk_size):
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    embeddings = []
    
    async def process_all_chunks():
        tasks = []
        for chunk in chunks:
            tasks.extend(await aget_embedding_chunk(chunk))
        return tasks
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    embeddings = loop.run_until_complete(process_all_chunks())
    loop.close()
    
    return embeddings


# embeddings = process_embeddings(contexts, 100)
# with open("embeddings.json", "w") as f:
#     json.dump(embeddings, f)