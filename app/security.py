import os
from fastapi import Header, HTTPException, status

AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "default-dev-token-change-me")

def verify_token(authorization: str = Header(None)):
    """
    Verifies that the request contains the correct Bearer token.
    The main backend must send: Authorization: Bearer <AI_SERVICE_TOKEN>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header."
        )
    
    token = authorization.split(" ")[1]
    if token != AI_SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid service token."
        )
    
    return True