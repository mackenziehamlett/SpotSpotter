from typing import Optional

from fastapi import FastAPI
from fastapi import File, UploadFile

app = FastAPI()


# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}


@app.post("/checkFile/")
async def create_upload_file(file: UploadFile):
    # todo use the model/algorithms to solve stuff
    return {"filename": file.filename}

app = FastAPI()

