from src.state import LocalDocument

from pathlib import Path

import pymupdf

directory = Path("./documents")

def load_local_documents(directory: Path) -> list[LocalDocument]:
    res = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".md", ".txt", ".pdf"]:
            continue

        content = extract_text(path) 
        res.append(LocalDocument(
            source_path=str(path),
            content=content,
        ))
    return res

def extract_text(path: Path) -> str:
    if path.suffix.lower() in [".md", ".txt"]:
         content = path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            content = "\n".join(page.get_text() for page in pdf)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return content
     

if __name__ == "__main__":
    print(load_local_documents(directory))
