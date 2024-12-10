import secrets
from typing import List
from pydantic import AnyHttpUrl, EmailStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    # SECRET_KEY for JWT token generation
    # Calling secrets.token_urlsafe will generate a new secret everytime
    # the server restarts, which can be quite annoying when developing, where
    # a stable SECRET_KEY is prefered.

    # SECRET_KEY: str = secrets.token_urlsafe(32)
    SECRET_KEY: str = "mhchatbotsecretkey"

    # database configurations
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 2
    SERVER_NAME: str
    SERVER_HOST: AnyHttpUrl
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    PROJECT_NAME: str

    FIRST_SUPERUSER: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    SUPABASE_URL: str
    SUPABASE_KEY: str

    OPENAI_API_KEY: str



    VITE_BACKEND_API_URL: AnyHttpUrl


    # SSO ID and Secrets
    # GOOGLE_CLIENT_ID: str = None
    # GOOGLE_CLIENT_SECRET: str = None
    # SSO_CALLBACK_HOSTNAME: str = None
    # SSO_LOGIN_CALLBACK_URL: str = None

    class Config:
        env_file = ".env"


settings = Settings()
