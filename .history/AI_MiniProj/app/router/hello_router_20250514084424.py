from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/api/hello")
def hello_world():
    return JSONResponse(content={"message": "Hello, World!"}) 