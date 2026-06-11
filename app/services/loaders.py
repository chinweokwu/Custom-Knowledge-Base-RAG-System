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
from app.core.ai_manager import ai_manager
from app.core.config import settings
import asyncio
from docling.document_converter import DocumentConverter

# Initialize Logger
logger = get_logger("loaders")

import re

class StructuralSplitter:
    """
    Advanced 'Logical Integrity' Splitter.
    Treats code blocks, lists, and tables as atomic, protected entities
    that maintain structural integrity during split operations.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _segment_text(self, text: str) -> List[dict]:
        lines = text.splitlines(keepends=True)
        segments = []
        
        current_type = None
        current_lines = []
        
        def get_indent(line: str) -> int:
            return len(line) - len(line.lstrip(' \t'))
            
        list_base_indent = 0
        code_base_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 1. Inside fenced code
            if current_type == "fenced_code":
                current_lines.append(line)
                if stripped.startswith("```"):
                    segments.append({"type": "code", "content": "".join(current_lines)})
                    current_type = None
                    current_lines = []
                i += 1
                continue
                
            # 2. Check for start of fenced code
            if stripped.startswith("```"):
                if current_type:
                    segments.append({"type": current_type, "content": "".join(current_lines)})
                current_type = "fenced_code"
                current_lines = [line]
                i += 1
                continue
                
            # 3. Inside table
            if current_type == "table":
                if (stripped.startswith("|") and stripped.endswith("|")) or (len(stripped) > 2 and stripped.count("|") >= 2):
                    current_lines.append(line)
                    i += 1
                    continue
                else:
                    segments.append({"type": "table", "content": "".join(current_lines)})
                    current_type = None
                    current_lines = []
                    continue
                    
            # 4. Check for table start
            if stripped.startswith("|") and (stripped.count("|") >= 2):
                if current_type:
                    segments.append({"type": current_type, "content": "".join(current_lines)})
                current_type = "table"
                current_lines = [line]
                i += 1
                continue
                
            # 5. Inside python/JS code block (def/class)
            if current_type == "code_block":
                indent = get_indent(line)
                if not stripped or indent > code_base_indent:
                    current_lines.append(line)
                    i += 1
                    continue
                else:
                    segments.append({"type": "code", "content": "".join(current_lines)})
                    current_type = None
                    current_lines = []
                    continue
                    
            # 6. Check for python/JS code start (def/class)
            if stripped.startswith(("def ", "class ")):
                if current_type:
                    segments.append({"type": current_type, "content": "".join(current_lines)})
                current_type = "code_block"
                code_base_indent = get_indent(line)
                current_lines = [line]
                i += 1
                continue
                
            # 7. Check for list item start
            list_match = re.match(r"^[ \t]*(?:\d+\.|\*|\-|\+)\s+", line)
            if list_match:
                if current_type and current_type != "list":
                    segments.append({"type": current_type, "content": "".join(current_lines)})
                    current_type = "list"
                    list_base_indent = get_indent(line)
                    current_lines = [line]
                elif not current_type:
                    current_type = "list"
                    list_base_indent = get_indent(line)
                    current_lines = [line]
                else:
                    current_lines.append(line)
                i += 1
                continue
                
            # 8. Inside list: check if it should continue
            if current_type == "list":
                indent = get_indent(line)
                if not stripped or indent > list_base_indent:
                    current_lines.append(line)
                    i += 1
                    continue
                else:
                    segments.append({"type": "list", "content": "".join(current_lines)})
                    current_type = None
                    current_lines = []
                    continue
                    
            # 9. Default: append to text paragraph
            if not current_type:
                current_type = "text"
                current_lines = [line]
            else:
                current_lines.append(line)
            i += 1
            
        if current_type:
            segments.append({"type": "code" if current_type == "code_block" else current_type, "content": "".join(current_lines)})
            
        return segments
    def _split_large_table(self, table_text: str, chunk_size: int) -> List[str]:
        lines = table_text.splitlines(keepends=True)
        if len(lines) <= 2:
            return [table_text]
        
        header_lines = lines[:2]
        header_text = "".join(header_lines)
        data_lines = lines[2:]
        
        sub_tables = []
        current_rows = []
        
        for row in data_lines:
            potential_len = len(header_text) + len("".join(current_rows + [row]))
            if current_rows and potential_len > chunk_size:
                sub_tables.append(header_text + "".join(current_rows))
                current_rows = [row]
            else:
                current_rows.append(row)
                
        if current_rows:
            sub_tables.append(header_text + "".join(current_rows))
            
        return sub_tables

    def _split_large_code(self, code_text: str, chunk_size: int) -> List[str]:
        lines = code_text.splitlines(keepends=True)
        if not lines:
            return [code_text]
            
        first_line = lines[0]
        last_line = lines[-1]
        
        is_fenced = first_line.startswith("```") and last_line.startswith("```")
        if not is_fenced:
            sub_blocks = []
            current_lines = []
            for line in lines:
                potential_len = len("".join(current_lines + [line]))
                if current_lines and potential_len > chunk_size:
                    sub_blocks.append("".join(current_lines))
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_lines:
                sub_blocks.append("".join(current_lines))
            return sub_blocks
            
        lang_tag = first_line
        closing_tag = "```\n"
        code_body_lines = lines[1:-1]
        
        sub_blocks = []
        current_lines = []
        
        for line in code_body_lines:
            potential_len = len(lang_tag) + len("".join(current_lines + [line])) + len(closing_tag)
            if current_lines and potential_len > chunk_size:
                sub_blocks.append(lang_tag + "".join(current_lines) + closing_tag)
                current_lines = [line]
            else:
                current_lines.append(line)
                
        if current_lines:
            sub_blocks.append(lang_tag + "".join(current_lines) + closing_tag)
            
        return sub_blocks

    def _get_atomic_pieces(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        segments = self._segment_text(text)
        atomic_pieces = []
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for seg in segments:
            if seg["type"] == "text":
                sub_chunks = text_splitter.split_text(seg["content"])
                atomic_pieces.extend(sub_chunks)
            else:
                content = seg["content"]
                if len(content) > max(chunk_size * 2, 2000):
                    if seg["type"] == "table":
                        sub_chunks = self._split_large_table(content, chunk_size)
                        atomic_pieces.extend(sub_chunks)
                    elif seg["type"] == "code":
                        sub_chunks = self._split_large_code(content, chunk_size)
                        atomic_pieces.extend(sub_chunks)
                    else:
                        fallback_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        sub_chunks = fallback_splitter.split_text(content)
                        atomic_pieces.extend(sub_chunks)
                else:
                    atomic_pieces.append(content)
                    
        return atomic_pieces

    def _merge_pieces(self, pieces: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        chunks = []
        current_chunk = []
        current_len = 0
        
        def join_pieces(parts: List[str]) -> str:
            res = ""
            for p in parts:
                if not res:
                    res = p
                else:
                    if res.endswith("\n") or p.startswith("\n"):
                        res += p
                    else:
                        res += "\n" + p
            return res
            
        for piece in pieces:
            if not piece.strip():
                continue
            
            merged_temp = join_pieces(current_chunk + [piece])
            if current_chunk and len(merged_temp) > chunk_size:
                chunks.append(join_pieces(current_chunk))
                
                # Rebuild overlap
                overlap_content = []
                for prev_piece in reversed(current_chunk):
                    temp_overlap = join_pieces([prev_piece] + overlap_content)
                    if len(temp_overlap) <= chunk_overlap:
                        overlap_content.insert(0, prev_piece)
                    else:
                        break
                current_chunk = overlap_content + [piece]
                current_len = len(join_pieces(current_chunk))
            else:
                current_chunk.append(piece)
                current_len = len(merged_temp)
                
        if current_chunk:
            chunks.append(join_pieces(current_chunk))
            
        return chunks

    def split_text(self, text: str, hierarchical: bool = False) -> List[str] | List[tuple]:
        if hierarchical:
            parent_chunk_size = 3000
            parent_chunk_overlap = 500
            parent_pieces = self._get_atomic_pieces(text, parent_chunk_size, parent_chunk_overlap)
            parents = self._merge_pieces(parent_pieces, parent_chunk_size, parent_chunk_overlap)
            
            final_hierarchy = []
            for parent in parents:
                child_pieces = self._get_atomic_pieces(parent, self.chunk_size, self.chunk_overlap)
                children = self._merge_pieces(child_pieces, self.chunk_size, self.chunk_overlap)
                for child in children:
                    final_hierarchy.append((child, parent))
            return final_hierarchy
            
        pieces = self._get_atomic_pieces(text, self.chunk_size, self.chunk_overlap)
        return self._merge_pieces(pieces, self.chunk_size, self.chunk_overlap)


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
    
    # 2. Handle PDF (Includes Vision & Structural Extraction)
    elif source.lower().endswith(".pdf"):
        raw_docs = []
        media_dir = settings.MEDIA_DIR
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        try:
            # --- PHASE 1: Structural Extraction via Docling ---
            logger.info("Using Docling for technical structural extraction.")
            converter = DocumentConverter()
            result = converter.convert(source)
            md_content = result.document.export_to_markdown()
            
            raw_docs.append(Document(
                page_content=md_content, 
                metadata={"source": source, "parser": "docling"}
            ))
            
            # --- PHASE 2: Image/Vision Extraction via PyMuPDF (Fitz) ---
            # Docling is great for text/tables, but we use Fitz for high-speed image harvesting
            import fitz
            doc = fitz.open(source)
            for i, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        timestamp = int(time.time())
                        clean_name = os.path.basename(source).replace(" ", "_")
                        img_filename = f"img_{timestamp}_{i}_{img_idx}_{clean_name}.jpg"
                        img_path = os.path.join(media_dir, img_filename)

                        # Save and Compress
                        with Image.open(io.BytesIO(image_bytes)) as pil_img:
                            if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
                            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            pil_img.save(img_path, "JPEG", quality=70, optimize=True)

                        # Vision Description
                        description = asyncio.run(ai_manager.describe_image(image_bytes))
                        visual_vector = asyncio.run(ai_manager.get_clip_embedding(Image.open(io.BytesIO(image_bytes))))

                        raw_docs.append(Document(
                            page_content=f"[VISUAL_ANALYSIS]: {description}",
                            metadata={
                                "source": source, "page": i + 1, "is_visual": True, 
                                "visual_embedding": visual_vector, "media_url": f"/media/{img_filename}",
                                "meaning_type": "image_content"
                            }
                        ))
                    except Exception as img_err:
                        logger.warning(f"Failed image extraction on page {i}: {img_err}")
            doc.close()
            
        except Exception as e:
            logger.error(f"Advanced PDF parsing failed: {e}. Falling back to basic text extraction.")
            # Basic fallback if everything else fails
            try:
                import fitz
                doc = fitz.open(source)
                for page in doc:
                    raw_docs.append(Document(page_content=page.get_text(), metadata={"source": source}))
                doc.close()
            except: pass
        
        return raw_docs

    # 3. Handle Word Docs (Includes Image Detection)
    elif source.lower().endswith(".docx"):
        logger.info("Detected Word source. Using Unstructured elements with context-aware merging.")
        base_loader = UnstructuredWordDocumentLoader(
            source, 
            mode="elements"
        )
        class MergedWordLoader:
            def __init__(self, raw_loader):
                self.raw_loader = raw_loader
            def load(self) -> List[Document]:
                elements = self.raw_loader.load()
                merged_docs = []
                current_text = []
                current_metadata = {}
                
                for el in elements:
                    content = el.page_content.strip()
                    if not content:
                        continue
                    
                    category = el.metadata.get("category", "")
                    is_special = category in ["Image", "Title", "Heading", "Table"]
                    
                    if is_special:
                        # Flush accumulated paragraph text
                        if current_text:
                            merged_docs.append(Document(
                                page_content="\n\n".join(current_text),
                                metadata=dict(current_metadata)
                            ))
                            current_text = []
                            current_metadata = {}
                        # Add special structural element directly
                        merged_docs.append(el)
                    else:
                        current_text.append(content)
                        if not current_metadata:
                            current_metadata = el.metadata
                        else:
                            if "page_number" in el.metadata:
                                current_metadata["page_number"] = el.metadata["page_number"]
                        
                        # Merge if accumulated text hits target size (800 chars)
                        if sum(len(t) for t in current_text) >= 800:
                            merged_docs.append(Document(
                                page_content="\n\n".join(current_text),
                                metadata=dict(current_metadata)
                            ))
                            current_text = []
                            current_metadata = {}
                
                if current_text:
                    merged_docs.append(Document(
                        page_content="\n\n".join(current_text),
                        metadata=dict(current_metadata)
                    ))
                return merged_docs
        loader = MergedWordLoader(base_loader)

    # 4. Handle CSV
    elif source.endswith(".csv"):
        logger.info("Detected CSV source. Using CSVLoader.")
        loader = CSVLoader(source)

    # 5. Handle Excel (Row-Aware for Structured Intelligence)
    elif source.endswith((".xlsx", ".xls")):
        logger.info("Detected Excel source. Using Row-Aware Pandas Loader with Dynamic Table Grouping.")
        class RowAwareLoader:
            def __init__(self, path): self.path = path
            def load(self):
                from openpyxl import load_workbook
                wb = load_workbook(self.path, data_only=True)
                docs = []
                
                # Scan for images/charts in all sheets and perform Vision Analysis
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    # Note: openpyxl stores images in ws._images
                    if hasattr(ws, '_images') and ws._images:
                        for img_idx, img in enumerate(ws._images):
                            try:
                                # 1. Extract raw bytes from openpyxl image
                                from io import BytesIO
                                img_data = img._data() # Accesses the raw image stream
                                
                                # 2. Save with Smart Shrinking (JPEG)
                                timestamp = int(time.time())
                                img_filename = f"excel_img_{timestamp}_{sheet_name}_{img_idx}.jpg"
                                img_path = os.path.join(media_dir, img_filename)
                                
                                with Image.open(BytesIO(img_data)) as pil_img:
                                    if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
                                    pil_img.save(img_path, "JPEG", quality=70)
                                
                                # 3. Vision Analysis
                                logger.info(f"Analyzing Excel visual on sheet '{sheet_name}'...")
                                description = asyncio.run(ai_manager.describe_image(img_data))
                                
                                docs.append(Document(
                                    page_content=f"[EXCEL_VISUAL_ANALYSIS from Sheet '{sheet_name}']: {description}",
                                    metadata={
                                        "source": self.path, 
                                        "sheet": sheet_name, 
                                        "is_visual": True,
                                        "media_url": f"/media/{img_filename}",
                                        "meaning_type": "spreadsheet_visual"
                                    }
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to extract Excel image: {e}")
                
                # Load data using Polars (Rust-backed high speed)
                df = pl.read_excel(self.path)
                headers = df.columns
                rows = df.to_dicts()
                
                # Dynamic Grouping based on Target Chunk Size (~1,000 characters)
                current_batch = []
                current_start_idx = 1
                current_char_count = 0
                max_chunk_chars = 1000
                
                def make_md_table(header_list, row_list):
                    md = "| " + " | ".join(header_list) + " |\n"
                    md += "| " + " | ".join(["---"] * len(header_list)) + " |\n"
                    for r in row_list:
                        row_vals = [str(r.get(h, "")) for h in header_list]
                        md += "| " + " | ".join(row_vals) + " |\n"
                    return md
                
                for idx, row in enumerate(rows):
                    row_vals = [str(row.get(h, "")) for h in headers]
                    row_char_len = len(" | ".join(row_vals)) + 4
                    
                    if current_batch and (current_char_count + row_char_len > max_chunk_chars):
                        # Flush current batch
                        md_table = make_md_table(headers, current_batch)
                        end_idx = idx
                        docs.append(Document(
                            page_content=f"Excel Data Fragment from {os.path.basename(self.path)} (Rows {current_start_idx} to {end_idx}):\n\n{md_table}",
                            metadata={"source": self.path, "rows": f"{current_start_idx}-{end_idx}", "meaning_type": "spreadsheet_data"}
                        ))
                        current_batch = []
                        current_char_count = 0
                        current_start_idx = idx + 1
                    
                    current_batch.append(row)
                    current_char_count += row_char_len
                
                if current_batch:
                    md_table = make_md_table(headers, current_batch)
                    end_idx = len(rows)
                    docs.append(Document(
                        page_content=f"Excel Data Fragment from {os.path.basename(self.path)} (Rows {current_start_idx} to {end_idx}):\n\n{md_table}",
                        metadata={"source": self.path, "rows": f"{current_start_idx}-{end_idx}", "meaning_type": "spreadsheet_data"}
                    ))
                return docs
        loader = RowAwareLoader(source)

    # 6. Handle JSON (Tree-Aware Custom Chunker)
    elif source.lower().endswith(".json"):
        logger.info("Detected JSON source. Using JSONStructureLoader.")
        class JSONStructureLoader:
            def __init__(self, path: str):
                self.path = path
            def load(self) -> List[Document]:
                import json
                from typing import Any
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                docs = []
                filename = os.path.basename(self.path)

                def serialize_and_create_doc(obj: Any, path_str: str) -> Document:
                    content = json.dumps(obj, indent=2, ensure_ascii=False)
                    return Document(
                        page_content=f"JSON Fragment from {filename} at path '{path_str}':\n\n{content}",
                        metadata={
                            "source": self.path,
                            "json_path": path_str,
                            "meaning_type": "structured_json"
                        }
                    )

                def traverse(obj: Any, current_path: str = "$"):
                    if isinstance(obj, dict):
                        serialized = json.dumps(obj, ensure_ascii=False)
                        if len(serialized) <= 1000:
                            docs.append(serialize_and_create_doc(obj, current_path))
                        else:
                            for k, v in obj.items():
                                new_path = f"{current_path}.{k}"
                                if isinstance(v, (dict, list)):
                                    traverse(v, new_path)
                                else:
                                    docs.append(serialize_and_create_doc({k: v}, current_path))
                    elif isinstance(obj, list):
                        serialized = json.dumps(obj, ensure_ascii=False)
                        if len(serialized) <= 1000:
                            docs.append(serialize_and_create_doc(obj, current_path))
                        else:
                            simple_elements = []
                            for idx, val in enumerate(obj):
                                if isinstance(val, (dict, list)):
                                    if simple_elements:
                                        docs.append(serialize_and_create_doc(simple_elements, f"{current_path}[{idx-len(simple_elements)}:{idx}]"))
                                        simple_elements = []
                                    traverse(val, f"{current_path}[{idx}]")
                                else:
                                    simple_elements.append(val)
                                    if len(json.dumps(simple_elements, ensure_ascii=False)) > 1000:
                                        docs.append(serialize_and_create_doc(simple_elements, f"{current_path}[{idx-len(simple_elements)+1}:{idx+1}]"))
                                        simple_elements = []
                            if simple_elements:
                                docs.append(serialize_and_create_doc(simple_elements, f"{current_path}[{len(obj)-len(simple_elements)}:{len(obj)}]"))
                    else:
                        docs.append(serialize_and_create_doc(obj, current_path))

                traverse(data)
                if not docs:
                    docs.append(serialize_and_create_doc(data, "$"))
                return docs
        loader = JSONStructureLoader(source)

    # 7. Default to Text
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