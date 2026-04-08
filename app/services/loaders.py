from typing import List
import os
import time
import polars as pl
from PIL import Image
import io
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPDFLoader
)
from langchain_core.documents import Document
from app.core.logger_config import get_logger

# Initialize Logger
logger = get_logger("loaders")

import re

class StructuralSplitter:
    """
    Advanced 'Logical Integrity' Splitter.
    Uses regex to recognize and preserve TQL code blocks, hierarchical lists, and tables.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Patterns for blocks that MUST stay together
        self.patterns = {
            "tql_code": re.compile(r"var\s+\w+\s*=\s*`.*?`|select\s+.*?\s+from\s+.*?(?:\s+where\s+.*?)?", re.DOTALL | re.IGNORECASE),
            "tech_list": re.compile(r"^(?:\d+\.|\*|\-)\s+.*?(?:\n\s+(?:\d+\.|\*|\-)\s+.*?)*", re.MULTILINE),
            "data_table": re.compile(r"^[|\-+]{3,}.*?^[|\-+]{3,}", re.DOTALL | re.MULTILINE)
        }

    def split_text(self, text: str, hierarchical: bool = False) -> List[str] | List[tuple]:
        """
        Splits text into chunks. 
        If hierarchical=True, returns a list of (child_chunk, parent_chunk) tuples.
        """
        # 1. Protect meaningful blocks by temporarily extracting them
        protected_blocks = []
        placeholder_template = "[[PROTECTED_BLOCK_{idx}]]"
        
        modified_text = text
        for name, pattern in self.patterns.items():
            matches = list(pattern.finditer(modified_text))
            # Sort matches backwards to avoid offset issues
            for match in reversed(matches):
                block_content = match.group(0)
                idx = len(protected_blocks)
                placeholder = placeholder_template.format(idx=idx)
                protected_blocks.append(block_content)
                modified_text = modified_text[:match.start()] + placeholder + modified_text[match.end():]

        if hierarchical:
            # First level: Large Parent Chunks
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, 
                chunk_overlap=500
            )
            parents = parent_splitter.split_text(modified_text)
            
            final_hierarchy = []
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            for p_idx, parent_text in enumerate(parents):
                # Restore parent placeholders
                restored_parent = parent_text
                for idx, block in enumerate(protected_blocks):
                    ph = placeholder_template.format(idx=idx)
                    if ph in restored_parent:
                        restored_parent = restored_parent.replace(ph, block)
                
                # Split parent into children
                children = child_splitter.split_text(parent_text)
                for child_text in children:
                    # Restore child placeholders
                    restored_child = child_text
                    for idx, block in enumerate(protected_blocks):
                        ph = placeholder_template.format(idx=idx)
                        if ph in restored_child:
                            restored_child = restored_child.replace(ph, block)
                    
                    final_hierarchy.append((restored_child, restored_parent))
            
            return final_hierarchy

        # 2. Split the remaining text normally (by paragraph/newline)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        raw_chunks = base_splitter.split_text(modified_text)

        # 3. Re-inject the protected blocks into the chunks
        final_chunks = []
        for chunk in raw_chunks:
            restored_chunk = chunk
            for idx, block in enumerate(protected_blocks):
                placeholder = placeholder_template.format(idx=idx)
                if placeholder in restored_chunk:
                    restored_chunk = restored_chunk.replace(placeholder, block)
            final_chunks.append(restored_chunk)
            
        return final_chunks

structural_splitter = StructuralSplitter()

def load_document(source: str, heavy_parsing: bool = False) -> List[Document]:
    """
    Smarter document loader that extracts text and identifies images/snapshots.
    Supports: .pdf, .docx, .txt, .csv, .xlsx, and URLs.
    """
    logger.info(f"Attempting to load document from source: {source} (Heavy: {heavy_parsing})")
    
    # 1. Handle URL Sources
    if source.startswith(("http://", "https://")):
        logger.info("Detected URL source. Using WebBaseLoader.")
        loader = WebBaseLoader(source)
    
    # 2. Handle PDF (Includes Soft Visual Discovery)
    elif source.lower().endswith(".pdf"):
        logger.info("Detected PDF source. Loading text and scanning for images...")
        loader = PyPDFLoader(source)
        raw_docs = loader.load()
        
        # Soft Visual Discovery: Scan and Extract images
        import pypdf
        reader = pypdf.PdfReader(source)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        media_dir = os.path.join(BASE_DIR, "media")
        
        for i, page in enumerate(reader.pages):
            # pypdf 3.0.0+ image extraction
            if page.images:
                for img_idx, image in enumerate(page.images):
                    try:
                        # Create unique filename
                        timestamp = int(time.time())
                        clean_name = os.path.basename(source).replace(" ", "_")
                        img_filename = f"img_{timestamp}_{i}_{img_idx}_{clean_name}.png"
                        img_path = os.path.join(media_dir, img_filename)
                        
                        # Save image data
                        with open(img_path, "wb") as f:
                            f.write(image.data)
                        
                        logger.info(f"Extracted image to: {img_path}")
                        
                        # Create a "Visual Chunk" for the Vector DB
                        visual_doc = Document(
                            page_content=f"[VISUAL_EVIDENCE: Visual element/image extracted from {os.path.basename(source)} on Page {i+1}]",
                            metadata={
                                "source": source, 
                                "page": i + 1, 
                                "is_visual": True, 
                                "media_url": f"/media/{img_filename}",
                                "meaning_type": "image_content"
                            }
                        )
                        raw_docs.append(visual_doc)
                    except Exception as img_err:
                        logger.warning(f"Failed to extract image {img_idx} on page {i}: {img_err}")
                
                # We limit to first image per page to avoid overwhelming the DB in this phase
                # break 
        
        return raw_docs

    # 3. Handle Word Docs (Includes Image Detection)
    elif source.lower().endswith(".docx"):
        logger.info("Detected Word source. Using Unstructured elements to detect images.")
        # Standard Unstructured identifies 'Image' elements without Tesseract
        loader = UnstructuredWordDocumentLoader(
            source, 
            mode="elements"
        )

    # 4. Handle CSV
    elif source.endswith(".csv"):
        logger.info("Detected CSV source. Using CSVLoader.")
        loader = CSVLoader(source)

    # 5. Handle Excel (Row-Aware for Structured Intelligence)
    elif source.endswith((".xlsx", ".xls")):
        logger.info("Detected Excel source. Using Row-Aware Pandas Loader.")
        class RowAwareLoader:
            def __init__(self, path): self.path = path
            def load(self):
                from openpyxl import load_workbook
                wb = load_workbook(self.path, data_only=True)
                docs = []
                
                # Scan for images/charts in all sheets
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    image_count = len(ws._images)
                    chart_count = len(ws._charts)
                    if image_count > 0 or chart_count > 0:
                        docs.append(Document(
                            page_content=f"[SOFT_VISUAL_DISCOVERY: Sheet '{sheet_name}' contains {image_count} images and {chart_count} charts]",
                            metadata={"source": self.path, "sheet": sheet_name, "is_visual": True}
                        ))

                # Load data using Polars (Rust-backed high speed)
                df = pl.read_excel(self.path)
                for idx, row in enumerate(df.to_dicts()):
                    text = " | ".join([f"[{k}]: {v}" for k, v in row.items() if v is not None])
                    docs.append(Document(page_content=text, metadata={"source": self.path, "row": idx}))
                return docs
        loader = RowAwareLoader(source)

    # 6. Default to Text
    else:
        if not os.path.exists(source):
             logger.error(f"File not found: {source}")
             raise ValueError(f"File not found: {source}")
        logger.info("Defaulting to TextLoader.")
        loader = TextLoader(source, encoding="utf-8")

    try:
        # Load the raw documents
        raw_docs = loader.load()
        
        # 7. Post-Processing: Assign "Meaning" to Image Chunks
        final_docs = []
        for doc in raw_docs:
            category = doc.metadata.get("category", "")
            
            # If the chunk is a snapshot or image but has no text content
            if category == "Image" or not doc.page_content.strip():
                filename = os.path.basename(source)
                # We give it descriptive text so it can be vectorized
                doc.page_content = f"Visual representation/snapshot from {filename}"
                doc.metadata["is_visual"] = True
                doc.metadata["meaning_type"] = "image_snapshot"

            final_docs.append(doc)

        logger.info(f"Successfully loaded {len(final_docs)} document sections.")
        return final_docs

    except Exception as e:
        logger.exception(f"Failed to load document from {source}: {e}")
        raise

def extract_chunks_from_source(source: str, heavy_parsing: bool = False, hierarchical: bool = True) -> List[str] | List[tuple]:
    """Returns discrete content blocks. If hierarchical=True, returns (child, parent) tuples."""
    try:
        docs = load_document(source, heavy_parsing)
        all_chunks = []
        for doc in docs:
            # Check if this doc needs further structural splitting
            if len(doc.page_content) > 500 and not doc.metadata.get("is_visual"):
                sub_chunks = structural_splitter.split_text(doc.page_content, hierarchical=hierarchical)
                all_chunks.extend(sub_chunks)
            else:
                if hierarchical:
                    # Visual chunks or very small chunks are their own parents
                    all_chunks.append((doc.page_content, doc.page_content))
                else:
                    all_chunks.append(doc.page_content)
                
        # Ensure only non-empty content is sent to the Vector DB
        if hierarchical:
            chunks = [c for c in all_chunks if c[0].strip()]
        else:
            chunks = [c for c in all_chunks if c.strip()]
        return chunks
    except Exception as e:
        logger.error(f"Chunk extraction failed: {e}")
        raise

def extract_text_from_source(source: str, heavy_parsing: bool = False) -> str:
    """Returns all text content from a source as a single string."""
    try:
        docs = load_document(source, heavy_parsing)
        # We still join with double newline, but structural_splitter will 
        # be used later in the API if needed for manual text.
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise