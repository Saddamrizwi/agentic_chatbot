import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from user.main import router as user_router   


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#user
app.include_router(user_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)