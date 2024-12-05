from datetime import timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic.networks import EmailStr
from ..auth import get_current_active_user


from ..models import User
from ..config import settings
from ..modules.chatbotllm import chatbotllm, fetch_all_history
from ..modules.semanticsearch import semanticsearch

router = APIRouter()

@router.post("/chatbotllm")
async def chatbot_llm(
    user_message: str = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    """
    Chat with Medical LLM (domain-specific Model)
    """
    response = chatbotllm(user_message, current_user.uuid)
    return {"response": response}

@router.get("/chathistory")
async def chat_history(
    current_user: User = Depends(get_current_active_user),
):
    """
    Chat with Medical LLM (domain-specific Model)
    """
    response = fetch_all_history(current_user.uuid)
    return {"response": response}


@router.post("/semanticsearch")
async def semantic_search(
    query: str = Body(...)
):
    """
    Semantic Search
    """
    response = await semanticsearch(query)
    return {"response": response}


@router.post("/mlclassification")
async def classification_ml(
    query: str = Body(...)
):
    """
    Classification using Machine Learning Model
    """
    pass


