from fastapi import FastAPI
from app.router.hello_router import router as hello_router

app = FastAPI()

app.include_router(hello_router) 