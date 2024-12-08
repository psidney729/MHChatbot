from fastapi import APIRouter

from . import login, users, solutions

api_router = APIRouter()
api_router.include_router(login.router, prefix="/login", tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(solutions.router, prefix="/modules", tags=["modules"])


@api_router.get("/")
async def root():
    return {"message": "Welcome to the MHChatbot Backend API!"}
