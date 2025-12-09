from typing import Any

class Paragraph:
    text: str

class Document:
    paragraphs: list[Paragraph]

    def __init__(self, docx_path: str | bytes | Any = ...) -> None: ...
