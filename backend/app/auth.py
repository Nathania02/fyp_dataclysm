from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings
from app.storage import UserStorage
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")
ph = PasswordHasher()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # return pwd_context.verify(plain_password, hashed_password)
    try:
        # ph.verify() will raise an exception if the password or hash is bad
        ph.verify(hashed_password, plain_password)
        return True
    
    except VerifyMismatchError:
        # The password was incorrect
        return False
    
    except InvalidHash:
        # The stored hash is not a valid Argon2 hash
        return False
    
    except Exception:
        # Other potential errors (e.g., argon2 config mismatch)
        return False

def get_password_hash(password: str) -> str:
    # return pwd_context.hash(password)
    return ph.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = UserStorage.get_by_email(email)
    if user is None:
        raise credentials_exception
    return user