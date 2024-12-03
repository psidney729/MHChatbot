from typing import List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic.networks import EmailStr
from beanie.exceptions import RevisionIdWasChanged

from ..auth.auth import (
    get_hashed_password,
    get_current_active_superuser,
    get_current_active_user,
)
from .. import schemas, models
from ..config.config import settings

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

engine = create_async_engine(
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
    echo=False,
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

router = APIRouter()


@router.post("", response_model=schemas.User)
async def register_user(
    password: str = Body(...),
    email: EmailStr = Body(...),
    first_name: str = Body(None),
    last_name: str = Body(None),
):
    """
    Register a new user.
    """
    hashed_password = get_hashed_password(password)
    user = models.User(
        email=email,
        hashed_password=hashed_password,
        first_name=first_name,
        last_name=last_name,
    )
    async with async_session() as session:
        session.add(user)
        try:
            await session.commit()
            return user
        except IntegrityError:
            await session.rollback()
            raise HTTPException(
                status_code=400,
                detail="User with that email already exists."
            )


@router.get("", response_model=List[schemas.User])
async def get_users(
    limit: Optional[int] = 10,
    offset: Optional[int] = 0,
    admin_user: models.User = Depends(get_current_active_superuser),
):
    async with async_session() as session:
        result = await session.execute(
            select(models.User)
            .offset(offset)
            .limit(limit)
        )
        users = result.scalars().all()
        return users


@router.get("/me", response_model=schemas.User)
async def get_profile(
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user


@router.patch("/me", response_model=schemas.User)
async def update_profile(
    updateschema: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Update current user.
    """
    async with async_session() as session:
        update_data = updateschema.model_dump(
            exclude={"is_active", "is_superuser"}, exclude_unset=True
        )
        
        if "password" in update_data:
            update_data["hashed_password"] = get_hashed_password(update_data["password"])
            del update_data["password"]
            
        try:
            stmt = (
                update(models.User)
                .where(models.User.uuid == current_user.uuid)
                .values(**update_data)
                .returning(models.User)
            )
            result = await session.execute(stmt)
            updated_user = result.scalar_one()
            await session.commit()
            return updated_user
            
        except IntegrityError:
            await session.rollback()
            raise HTTPException(
                status_code=400, 
                detail="User with that email already exists."
            )


@router.delete("/me", response_model=schemas.User)
async def delete_me(user: models.User = Depends(get_current_active_user)):
    async with async_session() as session:
        stmt = select(models.User).where(models.User.uuid == user.uuid)
        result = await session.execute(stmt)
        deluser = result.scalar_one()
        await session.delete(deluser)
        await session.commit()
        return user


@router.patch("/{userid}", response_model=schemas.User)
async def update_user(
    userid: UUID,
    update: schemas.UserUpdate,
    admin_user: models.User = Depends(get_current_active_superuser),
) -> Any:
    """
    Update a user.

    ** Restricted to superuser **

    Parameters
    ----------
    userid : UUID
        the user's UUID
    update : schemas.UserUpdate
        the update data
    current_user : models.User, optional
        the current superuser, by default Depends(get_current_active_superuser)
    """
    async with async_session() as session:
        result = await session.execute(
            select(models.User).where(models.User.uuid == userid)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
            
        update_data = update.model_dump(exclude_unset=True)
        if update.password is not None:
            update_data["hashed_password"] = get_hashed_password(update.password)
            del update_data["password"]
            
        try:
            stmt = (
                update(models.User)
                .where(models.User.uuid == userid)
                .values(**update_data)
                .returning(models.User)
            )
            result = await session.execute(stmt)
            updated_user = result.scalar_one()
            await session.commit()
            return updated_user
            
        except IntegrityError:
            await session.rollback()
            raise HTTPException(
                status_code=400,
                detail="User with that email already exists."
            )


@router.get("/{userid}", response_model=schemas.User)
async def get_user(
    userid: UUID, admin_user: models.User = Depends(get_current_active_superuser)
):
    """
    Get User Info

    ** Restricted to superuser **

    Parameters
    ----------
    userid : UUID
        the user's UUID

    Returns
    -------
    schemas.User
        User info
    """
    async with async_session() as session:
        result = await session.execute(
            select(models.User).where(models.User.uuid == userid)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
            
        return user


@router.delete("/{userid}", response_model=schemas.User)
async def delete_user(
    userid: UUID, admin_user: models.User = Depends(get_current_active_superuser)
):
    async with async_session() as session:
        stmt = select(models.User).where(models.User.uuid == userid)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
            
        await session.delete(user)
        await session.commit()
        return user
