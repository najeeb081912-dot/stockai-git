"""Sign-in endpoints. Later features use get_current_user to know who is calling."""

import hmac
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

import userstore

# Optional: if set, people must enter this code to create an account.
SIGNUP_CODE = os.getenv("SIGNUP_CODE", "")

userstore.init()

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str
    password: str
    code: str = ""


def _bearer(authorization):
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def get_current_user(authorization: str | None = Header(default=None)):
    user = userstore.user_for_token(_bearer(authorization))

    if not user:
        raise HTTPException(status_code=401, detail="Please sign in")

    return user


@router.post("/register")
def register(body: Credentials):
    if SIGNUP_CODE and not hmac.compare_digest(body.code, SIGNUP_CODE):
        raise HTTPException(status_code=403, detail="Invalid sign-up code")

    try:
        user = userstore.create_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"token": userstore.create_session(user["id"]), "user": user}


@router.post("/login")
def login(body: Credentials):
    try:
        user = userstore.authenticate(body.username, body.password)
    except PermissionError as e:
        raise HTTPException(status_code=429, detail=str(e))

    if not user:
        raise HTTPException(status_code=401, detail="Wrong username or password")

    return {"token": userstore.create_session(user["id"]), "user": user}


@router.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    token = _bearer(authorization)

    if token:
        userstore.delete_session(token)

    return {"status": "ok"}


@router.get("/me")
def me(authorization: str | None = Header(default=None)):
    return {"user": get_current_user(authorization)}
