from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from .app.auth import get_hashed_password
from .app.config import settings
from .app.routers import api_router
from .app.models import User, History

engine = create_async_engine(
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
    echo=False,
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup PostgreSQL


    async with engine.begin() as conn:
        await conn.run_sync(User.metadata.create_all)
        await conn.run_sync(History.metadata.create_all)
        
    
    app.session_maker = async_session

    # Create initial superuser
    async with async_session() as session:
        try:
            user = await session.execute(select(User).filter(User.email == settings.FIRST_SUPERUSER))
            user = user.scalar_one_or_none()

            if not user:
                user = User(
                    email=settings.FIRST_SUPERUSER,
                    hashed_password=get_hashed_password(settings.FIRST_SUPERUSER_PASSWORD),
                    is_superuser=True,
                    is_active=True  # Explicitly set is_active
                )
                session.add(user)
                await session.commit()
                
        except Exception as e:
            print(f"Error creating superuser: {str(e)}")
            await session.rollback()
            raise
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            # See https://github.com/pydantic/pydantic/issues/7186 for reason of using rstrip
            str(origin).rstrip("/")
            for origin in settings.BACKEND_CORS_ORIGINS
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(api_router, prefix=settings.API_V1_STR)
