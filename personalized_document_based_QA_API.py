
from QAModel.DataModel import QuestionAnswerRequest
from QAModel.Question_answering import question_answering
from preprocessing.preprocessing import get_last_file_extension, test_embed, parse_document_contents, parse_text_from_docx, parse_text_from_pdf, text_to_docs
from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from starlette.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi import FastAPI, Form, Request, Response, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
from pathlib import Path
import os
import sys
from preprocessing import *
from QAModel import *
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY_SK')
if api_key is None:
    raise ValueError("The API key is not set in the environment variables.")

cur_pth = os.getcwd()
pth = os.path.dirname(os.path.realpath(cur_pth))

if cur_pth not in sys.path:
    sys.path.append(cur_pth)
log_format = '%(asctime)s:%(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")
QA_app = FastAPI(
    title="personalized memory based QA",
    description="This simple API utilizes openapi gpt 3.5 turbo. The aim of the API is to enable to ask questions to get precise information from any document they have on hand. Our motivation is the complexity of getting precise information from our folders or documents .",
    version="1.0.0",
)
templates = Jinja2Templates(directory="static")
QA_app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent.absolute() / "static"), name="static")


@QA_app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = {
        "request": request,
        "image_url_1": "/static/QA.png",
    }
    return templates.TemplateResponse("index.html", context)

# Endpoint to upload PDF and perform embeddings


@QA_app.post("/upload_doc/")
async def upload_doc(file_data: UploadFile = File(...)):
    content = await file_data.read()
    file_extension = get_last_file_extension(file_data.filename)
    pages = parse_document_contents(content, file_extension)
    index = test_embed(api_key, pages)
    return {"message": "PDF uploaded and embeddings done."}

# Endpoint to perform question answering


@QA_app.post("/personalized_doc_based_QA/", response_model=dict)
async def personalized_document_based_QA(question: str = Form(...), file_data: UploadFile = File(...)):

    content = await file_data.read()
    file_extension = get_last_file_extension(file_data.filename)
    pages = parse_document_contents(content, file_extension)
    documents = text_to_docs(pages)
    index = test_embed(api_key, documents)
    res = question_answering(api_key, question, index)
    return {"answer": res}

if __name__ == "__main__":
    uvicorn.run(QA_app, host="0.0.0.0", port=8000)
