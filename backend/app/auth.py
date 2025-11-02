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
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)
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

def get_token_from_header_or_query(request: Request, token_from_header: str = Depends(oauth2_scheme_optional)) -> str:
    if token_from_header:
        return token_from_header

    # If header is missing, check query parameters
    token_from_query = request.query_params.get("token")
    if not token_from_query:
        # If it's not in the header OR query params, raise the 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # If we found it in the query, return that
    return token_from_query

async def get_current_user(token: str = Depends(get_token_from_header_or_query)) -> dict:
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