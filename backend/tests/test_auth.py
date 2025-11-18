"""
Unit tests for authentication functionality
"""
import pytest
from fastapi import HTTPException
from app.auth import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    get_current_user
)
from app.storage import UserStorage
from jose import jwt
from app.config import settings

class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test that password hashing works correctly"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_password_verification_success(self):
        """Test successful password verification"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test failed password verification"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes"""
        password1 = "password1"
        password2 = "password2"
        
        hash1 = get_password_hash(password1)
        hash2 = get_password_hash(password2)
        
        assert hash1 != hash2
    
    def test_verify_password_with_invalid_hash(self):
        """Test password verification with an invalid hash format"""
        password = "testpassword123"
        invalid_hash = "not_a_valid_argon2_hash"
        
        # Should return False instead of raising exception
        assert verify_password(password, invalid_hash) is False
    
    def test_verify_password_with_malformed_hash(self):
        """Test password verification with a malformed hash"""
        password = "testpassword123"
        malformed_hash = "$argon2$invalid$format"
        
        # Should return False instead of raising exception
        assert verify_password(password, malformed_hash) is False
    
    def test_verify_password_with_empty_hash(self):
        """Test password verification with empty hash"""
        password = "testpassword123"
        empty_hash = ""
        
        # Should return False instead of raising exception
        assert verify_password(password, empty_hash) is False
    
    def test_verify_password_with_none_hash(self):
        """Test password verification with None hash"""
        password = "testpassword123"
        
        # Should return False instead of raising exception
        assert verify_password(password, None) is False


class TestTokenCreation:
    """Test JWT token creation and validation"""
    
    def test_create_access_token(self):
        """Test token creation"""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_contains_correct_data(self):
        """Test that token contains the correct user data"""
        email = "test@example.com"
        data = {"sub": email}
        token = create_access_token(data)
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["sub"] == email
        assert "exp" in payload
    
    def test_token_expiration(self):
        """Test that token has expiration time"""
        from datetime import timedelta
        data = {"sub": "test@example.com"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert "exp" in payload


class TestGetCurrentUser:
    """Test getting current user from token"""
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self, test_storage_files, test_user):
        """Test successful user retrieval"""
        token = create_access_token({"sub": test_user['email']})
        
        user = await get_current_user(token)
        
        assert user['email'] == test_user['email']
        assert user['id'] == test_user['id']
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, test_storage_files):
        """Test user retrieval with invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(invalid_token)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_current_user_nonexistent_user(self, test_storage_files):
        """Test user retrieval for non-existent user"""
        token = create_access_token({"sub": "nonexistent@example.com"})
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token)
        
        assert exc_info.value.status_code == 401
