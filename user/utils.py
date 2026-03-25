import os
import json
import tempfile
import shutil
import hashlib
from typing import List

from fastapi import UploadFile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# CONFIG 
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

FILE_RECORD_PATH = os.path.join(BASE_DIR, "file_record", "file_record.json")
PERSIST_DIR = os.path.join(BASE_DIR, "DB")
NAMESPACE = "public"
OCR_LANG = "eng"

# -----------------------------
# EMBEDDINGS
# -----------------------------

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": False}
        )
    return _embeddings

# -----------------------------
# TEXT SPLITTER
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# -----------------------------
# FILE RECORD HANDLING
# -----------------------------

def load_file_records():
    if not os.path.exists(FILE_RECORD_PATH):
        return {}
    with open(FILE_RECORD_PATH, "r") as f:
        return json.load(f)


def save_file_records(records):
    os.makedirs(os.path.dirname(FILE_RECORD_PATH), exist_ok=True)
    with open(FILE_RECORD_PATH, "w") as f:
        json.dump(records, f, indent=4)


# -----------------------------
# HASH (FOR DUPLICATE DETECTION)
# -----------------------------

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdf(path):
    texts = []
    pdfreader = PdfReader(path)

    for page_number, page in enumerate(pdfreader.pages):
        content = page.extract_text()

        if content:
            texts.append(content)
        else:
            images = convert_from_path(
                path,
                first_page=page_number + 1,
                last_page=page_number + 1,
            )
            for image in images:
                extracted = pytesseract.image_to_string(image, lang=OCR_LANG)
                if extracted.strip():
                    texts.append(extracted)

    return texts


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]


# -----------------------------
# FILE UPLOADING FUNCTION
# -----------------------------

def upload_files(files_list: List[UploadFile]):
    files_record = load_file_records()

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(), 
        collection_name=NAMESPACE
    )

    added = skipped = 0
    messages = []
    total = vectordb._collection.count()

    print(f"Initial DB count: {total}")

    all_chunks = []

    for file_obj in files_list:
        file_name = file_obj.filename
        tmp_path = None

        try:
            suffix = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file_obj.file, tmp)
                tmp_path = tmp.name
            
            file_hash = get_file_hash(tmp_path)

            if file_hash in files_record:
                skipped += files_record[file_hash]["chunks"]
                msg = f"{file_name} already exists (duplicate skipped)"
                print(msg)
                messages.append(msg)
                continue

            if file_name.endswith(".pdf"):
                texts = extract_text_from_pdf(tmp_path)
            elif file_name.endswith(".txt"):
                texts = extract_text_from_txt(tmp_path)
            else:
                msg = f"{file_name} unsupported format"
                print(msg)
                messages.append(msg)
                continue

            chunks = []
            for t in texts:
                for chunk in splitter.split_text(t):
                    chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": file_name}
                        )
                    )

            if not chunks:
                msg = f"{file_name} has no readable content"
                print(msg)
                messages.append(msg)
                continue

            all_chunks.extend(chunks)

            files_record[file_hash] = {
                "filename": file_name,
                "chunks": len(chunks)
            }

            added += len(chunks)
            msg = f"{file_name} processed successfully"
            print(msg)
            messages.append(msg)

        except Exception as e:
            err = f"Error processing {file_name}: {str(e)}"
            print(err)
            messages.append(err)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if all_chunks:
        vectordb.add_documents(all_chunks)
        # vectordb.persist()

    save_file_records(files_record)

    total = vectordb._collection.count()

    return {
        "Added": added,
        "Skipped": skipped,
        "Total": total,
        "Messages": messages
    }