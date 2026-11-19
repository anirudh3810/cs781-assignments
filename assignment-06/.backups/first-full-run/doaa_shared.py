"""Small transport-neutral core shared by the assignment notebook and MCP server.

No model is loaded at import time. The notebook owns all generation and graded
interfaces. This module owns validated artifacts, source extraction and retrieval.
"""
from __future__ import annotations

import hashlib
import ast
import io
import json
import logging
import math
import os
import re
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

VERSION = "a6-core-1"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BASE_URL = "https://www.iitk.ac.in/doaa/"
CRITERIA = ("relevance", "faithfulness", "completeness")
HTML_PARSER_VERSION = 2
LOG = logging.getLogger("a6")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stable_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    data = value if isinstance(value, bytes) else (value if isinstance(value, str) else stable_json(value)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def notebook_function_key(path, names):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:  # IPython magic cells contain no selected definitions
            continue
        definitions.extend(ast.dump(node, include_attributes=False) for node in tree.body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names)
    if not definitions:
        raise ValueError("No requested notebook implementation definitions found")
    return digest(definitions)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def ensure_model_snapshot(model_id, root, revision=None):
    """Local-dir downloads avoid Windows symlink privileges; weights remain exact."""
    folder = Path(root) / model_id.replace("/", "--")
    manifest_path = folder / "download_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["model_id"] == model_id and (not revision or manifest["revision"] == revision):
                if all((folder / name).is_file() and digest((folder / name).read_bytes()) == expected
                       for name, expected in manifest["files"].items()) and any(name.endswith(".safetensors") for name in manifest["files"]):
                    return str(folder), manifest["revision"]
        except (OSError, ValueError, KeyError):
            pass
    from huggingface_hub import HfApi, snapshot_download
    resolved_revision = revision or HfApi().model_info(model_id).sha
    snapshot_download(model_id, revision=resolved_revision, local_dir=folder,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], max_workers=2)
    files = {str(path.relative_to(folder)).replace("\\", "/"): digest(path.read_bytes())
             for path in folder.rglob("*") if path.is_file() and ".cache" not in path.parts
             and path.name != "download_manifest.json"}
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"No model weights downloaded for {model_id}")
    atomic_json(manifest_path, {"model_id": model_id, "revision": resolved_revision, "files": files, "created_at": utc_now()})
    return str(folder), resolved_revision


class ArtifactStore:
    """Checksummed JSON envelopes; invalid records are reported and recomputed."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, family, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", family) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Artifact family/key must be safe path components")
        return self.root / family / (key + ".json")

    def save(self, family, key, payload):
        envelope = {"schema": 1, "producer": VERSION, "key": key, "created_at": utc_now(),
                    "payload_sha256": digest(payload), "payload": payload}
        atomic_json(self.path(family, key), envelope)
        return payload

    def load(self, family, key):
        path = self.path(family, key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if (envelope["schema"] != 1 or envelope["producer"] != VERSION or envelope["key"] != key
                    or envelope["payload_sha256"] != digest(envelope["payload"])):
                raise ValueError("schema, producer, key or checksum mismatch")
            return envelope["payload"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Ignoring invalid checkpoint %s: %s", path, exc)
            return None

    def save_blob(self, family, key, data, details=None):
        binary = self.path(family, key).with_suffix(".bin")
        # Data is immutable by content hash; publish the checked descriptor last.
        binary = binary.with_name(key + "-" + digest(data)[:16] + ".bin")
        atomic_bytes(binary, data)
        self.save(family, key, {"file": binary.name, "sha256": digest(data), "details": details or {}})

    def load_blob(self, family, key):
        record = self.load(family, key)
        if record is None:
            return None
        try:
            name = record["file"]
            if Path(name).name != name:
                raise ValueError("Invalid artifact filename")
            data = (self.path(family, key).parent / name).read_bytes()
            if digest(data) != record["sha256"]:
                raise ValueError("Binary checksum mismatch")
            return data
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Ignoring invalid binary %s/%s: %s", family, key, exc)
            return None


def chunk_text_core(text, chunk_size):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)]


def clean_text(text):
    text = unicodedata.normalize("NFC", text or "").replace("\xa0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()


def normalize_doaa_url(link, base_url=BASE_URL):
    if not isinstance(link, str) or not link.strip() or link.strip().startswith("#"):
        return None
    try:
        parts = urlsplit(urljoin(base_url, link.strip()))
        if parts.scheme.lower() not in ("http", "https") or parts.username or parts.password:
            return None
        host = (parts.hostname or "").lower()
        if host not in ("iitk.ac.in", "www.iitk.ac.in"):
            return None
        if parts.path != "/doaa" and not parts.path.startswith("/doaa/"):
            return None
        port = parts.port
        # The live crawl verified 82 identical-body, same-path resources across
        # the two DOAA host aliases. Canonicalize them to avoid duplicate crawls.
        host = "www.iitk.ac.in"
        netloc = host + ((":" + str(port)) if port and port not in (80, 443) else "")
        # All actual requests use verified HTTPS: HTTP is a published legacy alias.
        return urlunsplit(("https", netloc, parts.path or "/", parts.query, ""))
    except (ValueError, TypeError):
        return None


def extract_html(body, url):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(body, "html.parser")
    links = sorted({value for anchor in soup.find_all("a", href=True)
                    if (value := normalize_doaa_url(anchor["href"], url))})
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    for element in soup.select("script,style,noscript,nav,footer,#top-menu-container,#header,#footer,.breadcrumbs,.breadcrumb"):
        element.decompose()
    content = soup.select_one(".item-page, .com-content-article__body, article, main, #main, #content")
    content = content or soup.body or soup
    text = clean_text(content.get_text("\n", strip=True))
    return {"title": title, "pages": [{"page": None, "text": text}] if text else [],
            "links": links, "warnings": [], "html_parser": HTML_PARSER_VERSION}


def extract_pdf(body, smoke=False):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(body), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("Encrypted PDF cannot be opened with an empty password")
    pages, warnings = [], []
    selected = list(range(min(len(reader.pages), 2))) if smoke else range(len(reader.pages))
    for number in selected:
        try:
            page = reader.pages[number]
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 64 * 1024 * 1024:
                raise ValueError("PDF page content stream exceeds 64 MiB parsing guard")
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"page": number + 1, "text": text})
            else:
                warnings.append(f"Page {number + 1}: no extractable text (possibly scanned)")
        except Exception as exc:
            warnings.append(f"Page {number + 1}: {type(exc).__name__}: {exc}")
    if not pages:
        raise ValueError("No extractable PDF text; OCR is not part of the baseline")
    return {"title": str((reader.metadata or {}).get("/Title", "PDF document")),
            "pages": pages, "links": [], "warnings": warnings, "total_pdf_pages": len(reader.pages)}


def download_resource(url, store, headers=None, refresh=False):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    cache_key = digest({"url": url, "download": 1})
    cached = None if refresh else store.load("downloads", cache_key)
    if cached:
        body = store.load_blob("raw", cached["body_hash"])
        if body is not None:
            return {**cached, "body": body, "cache_hit": True}
    with requests.Session() as session:
        retry = Retry(total=2, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        # The live site still publishes HTTP links although its port 80 refused
        # connections during verification. Use the same path over verified TLS.
        request_url = re.sub(r"^http://", "https://", url)
        with session.get(request_url, headers=headers or {"User-Agent": "Mozilla/5.0"},
                         timeout=(10, 45), stream=True) as response:
            response.raise_for_status()
            final_url = normalize_doaa_url(response.url)
            if final_url is None:
                raise ValueError(f"Redirect left DOAA scope: {response.url}")
            pieces, size = [], 0
            for block in response.iter_content(65536):
                size += len(block)
                if size > 50 * 1024 * 1024:
                    raise ValueError("Resource exceeds 50 MiB download guard")
                pieces.append(block)
            body = b"".join(pieces)
            content_type = response.headers.get("Content-Type", "").lower()
            pdf = body.lstrip().startswith(b"%PDF-")
            if ("pdf" in content_type or urlsplit(final_url).path.lower().endswith(".pdf")) and not pdf:
                raise ValueError("Expected PDF but response has no PDF signature")
            if not pdf and "html" not in content_type and not re.search(br"(?i)<(?:!doctype|html|body)", body[:4096]):
                raise ValueError(f"Unsupported content type {content_type!r}")
            metadata = {"url": url, "final_url": final_url, "response_url": response.url, "kind": "pdf" if pdf else "html",
                        "content_type": content_type, "body_hash": digest(body), "bytes": size,
                        "fetched_at": utc_now(), "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified")}
    store.save_blob("raw", metadata["body_hash"], body, metadata)
    store.save("downloads", cache_key, metadata)
    return {**metadata, "body": body, "cache_hit": False}


def crawl_doaa(base_url, store, headers=None, mode="full", refresh=False, max_urls=1500, workers=3):
    base = normalize_doaa_url(base_url)
    if base is None:
        raise ValueError("base_url must be an HTTP(S) URL within iitk.ac.in/doaa")
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    scope_key = digest({"base": base, "mode": mode, "version": VERSION})
    cached = None if refresh else store.load("corpus", scope_key)
    if cached and cached.get("html_parser") == HTML_PARSER_VERSION and cached.get("status") in ("complete", "complete_with_source_failures", "smoke"):
        if digest(cached["text"]) == cached["corpus_hash"]:
            print(f"Corpus cache hit: {len(cached['documents'])} documents; {cached['status']}")
            return cached
    state = None if refresh else store.load("crawl", scope_key)
    state = state or {"pending": [base], "done": [], "documents": {}, "failures": {}, "urls": {}}
    for saved_url, reference in state.get("document_refs", {}).items():
        document = store.load("crawl_documents", reference)
        if document is not None:
            state["documents"][saved_url] = document
        else:
            state["done"] = [u for u in state["done"] if u != saved_url]
            state["pending"].append(saved_url)
    # Migrate an earlier uncanonicalized frontier without losing its downloads.
    migrated = {}
    for old_url, doc in state["documents"].items():
        url = normalize_doaa_url(old_url)
        if url:
            previous = migrated.get(url)
            if previous is None or doc.get("fetched_at", "") > previous.get("fetched_at", ""):
                migrated[url] = doc
    state["documents"] = migrated
    for saved_url, doc in list(state["documents"].items()):
        if doc["kind"] == "html" and doc.get("html_parser") != HTML_PARSER_VERSION:
            body = store.load_blob("raw", doc["body_hash"])
            if body is not None:
                state["documents"][saved_url] = {**doc, **extract_html(body, doc["final_url"])}
            else:
                state["done"] = [u for u in state["done"] if normalize_doaa_url(u) != saved_url]
                state["pending"].append(saved_url)
    state["done"] = sorted({value for url in state["done"] if (value := normalize_doaa_url(url))})
    state["pending"] = sorted({value for url in state["pending"] if (value := normalize_doaa_url(url))} - set(state["done"]))
    state["inflight"] = sorted({value for url in state.get("inflight", []) if (value := normalize_doaa_url(url))})
    state["failures"] = {normalize_doaa_url(url): message for url, message in state["failures"].items()
                          if normalize_doaa_url(url) and (normalize_doaa_url(url) not in migrated or migrated[normalize_doaa_url(url)].get("warnings"))}
    state["document_refs"] = {}
    recovered = set(state.get("inflight", [])) - set(state["done"])
    http_retry = {u for u, error in state["failures"].items() if "port=80" in error}
    state["done"] = [u for u in state["done"] if u not in http_retry]
    state["pending"] = sorted(set(state["pending"]) | recovered | http_retry)
    state["inflight"] = []
    known_pdf = {doc["body_hash"]: {k: doc[k] for k in ("title", "pages", "links", "warnings")}
                 for doc in state["documents"].values() if doc["kind"] == "pdf"}

    def checkpoint():
        # Persist source text once; frontier updates must not rewrite every PDF.
        refs = state.setdefault("document_refs", {})
        for saved_url, document in state["documents"].items():
            reference = digest({"url": saved_url, "body": document["body_hash"], "mode": mode, "parser": VERSION,
                                "html_parser": document.get("html_parser")})
            if refs.get(saved_url) != reference:
                store.save("crawl_documents", reference, document)
                refs[saved_url] = reference
        store.save("crawl", scope_key, {**state, "documents": {}})

    def fetch_parse(url):
        resource = download_resource(url, store, headers, refresh=refresh)
        parse_key = digest({"body": resource["body_hash"], "kind": resource["kind"],
                            "parser": VERSION, "html_parser": HTML_PARSER_VERSION,
                            "smoke": mode == "smoke", "url": resource["final_url"] if resource["kind"] == "html" else None})
        parsed = store.load("documents", parse_key)
        if parsed is None and resource["kind"] == "pdf":
            parsed = known_pdf.get(resource["body_hash"])
        if parsed is None:
            parsed = extract_pdf(resource["body"], mode == "smoke") if resource["kind"] == "pdf" else extract_html(resource["body"], resource["final_url"])
            store.save("documents", parse_key, parsed)
        if resource["kind"] == "pdf":
            known_pdf[resource["body_hash"]] = parsed
        return {**{k: v for k, v in resource.items() if k != "body"}, **parsed}

    # A real homepage and linked-PDF sample must parse before expanding the crawl.
    if base not in state["done"]:
        root = fetch_parse(base)
        if not root["pages"]:
            raise RuntimeError("Homepage returned no usable text; corpus was not published")
        state["documents"][base] = root
        state["done"].append(base)
        state["pending"] = sorted(set(root["links"]) - set(state["done"]))
        checkpoint()
    pdf_links = [u for u in state["pending"] if urlsplit(u).path.lower().endswith(".pdf")]
    if pdf_links and not any(d["kind"] == "pdf" for d in state["documents"].values()):
        sample_error = []
        for pdf_url in pdf_links[:3]:
            try:
                doc = fetch_parse(pdf_url)
                state["documents"][pdf_url] = doc
                state["done"].append(pdf_url)
                state["pending"].remove(pdf_url)
                checkpoint()
                break
            except Exception as exc:
                sample_error.append(f"{pdf_url}: {exc}")
        else:
            raise RuntimeError("PDF smoke test failed; full crawl stopped: " + "; ".join(sample_error))
    limit = 3 if mode == "smoke" else max_urls
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        while state["pending"] and len(state["done"]) < limit:
            batch = state["pending"][:min(workers, limit - len(state["done"]))]
            state["pending"] = state["pending"][len(batch):]
            state["inflight"] = batch.copy()
            checkpoint()
            futures = [(url, pool.submit(fetch_parse, url)) for url in batch]
            for url, future in futures:  # fixed processing order despite parallel I/O
                try:
                    document = future.result()
                    state["documents"][url] = document
                    if document.get("warnings"):
                        state["failures"][url] = "; ".join(document["warnings"])
                    else:
                        state["failures"].pop(url, None)
                    new_links = set(document["links"]) - set(state["done"]) - set(batch)
                    state["pending"] = sorted(set(state["pending"]) | new_links)
                except Exception as exc:
                    state["failures"][url] = f"{type(exc).__name__}: {exc}"
                    LOG.warning("Source extraction failed: %s: %s", url, exc)
                state["done"].append(url)
                state["inflight"].remove(url)
                checkpoint()
                print(f"Crawl: {len(state['done'])} URLs processed, {len(state['pending'])} pending", flush=True)
    if state["pending"] and mode == "full":
        raise RuntimeError(f"Crawl incomplete at max_urls={max_urls}; {len(state['pending'])} URLs remain. Increase A6_MAX_URLS and rerun to resume.")
    documents, content_seen = [], {}
    for url, document in sorted(state["documents"].items()):
        combined = "\n\n".join(p["text"] for p in document["pages"])
        content_hash = digest(combined)
        if not combined:
            continue
        if content_hash in content_seen:
            content_seen[content_hash]["aliases"].append(url)
            continue
        entry = {**document, "aliases": [url], "text_hash": content_hash}
        content_seen[content_hash] = entry
        documents.append(entry)
    segments, spans, offset = [], [], 0
    for document in documents:
        for page in document["pages"]:
            if segments:
                segments.append("\n\n")
                offset += 2
            start = offset
            segments.append(page["text"])
            offset += len(page["text"])
            spans.append({"start": start, "end": offset, "url": document["final_url"],
                          "title": document["title"], "page": page["page"], "kind": document["kind"]})
    text = "".join(segments)
    if not text.strip():
        raise RuntimeError("No genuine source text was extracted; no corpus checkpoint published")
    result = {"status": "smoke" if mode == "smoke" else ("complete_with_source_failures" if state["failures"] else "complete"),
              "text": text, "corpus_hash": digest(text), "documents": documents, "spans": spans,
              "visited": sorted(set(state["done"])), "failures": state["failures"],
              "base_url": base, "mode": mode, "html_parser": HTML_PARSER_VERSION, "created_at": utc_now()}
    store.save("corpus", scope_key, result)
    store.save("active", "corpus_" + mode, {"scope_key": scope_key, "corpus_hash": result["corpus_hash"]})
    return result


def chunk_metadata_core(chunks, spans):
    metadata, start = [], 0
    ordered = sorted(spans, key=lambda span: span["start"])
    cursor = 0
    for number, text in enumerate(chunks):
        end = start + len(text)
        while cursor < len(ordered) and ordered[cursor]["end"] <= start:
            cursor += 1
        stop = cursor
        while stop < len(ordered) and ordered[stop]["start"] < end:
            stop += 1
        sources = [{**span, "chunk_start": max(start, span["start"]) - start,
                    "chunk_end": min(end, span["end"]) - start}
                   for span in ordered[cursor:stop] if span["end"] > start]
        metadata.append({"chunk_id": number, "start": start, "end": end, "sha256": digest(text), "sources": sources})
        start = end
    return metadata


def normalized_f32(vectors, dimension=None):
    import numpy as np
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or not matrix.shape[1] or (dimension and matrix.shape[1] != dimension):
        raise ValueError(f"Embedding dimension mismatch: {matrix.shape}, expected {dimension}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding contains NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Embedding contains zero-norm rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def encode_batches(chunks, model, store, batch_size=16, shard_size=128, model_key=None):
    import numpy as np
    if not chunks:
        raise ValueError("Cannot embed an empty corpus")
    config = {"model": EMBEDDING_MODEL, "model_key": model_key, "chunks": [digest(x) for x in chunks],
              "normalized": True, "dtype": "float32", "version": VERSION}
    key = digest(config)
    parts = []
    for start in range(0, len(chunks), shard_size):
        rows = chunks[start:start + shard_size]
        part_key = digest({"key": key, "start": start, "count": len(rows)})
        cached = store.load_blob("embedding_shards", part_key)
        matrix = None
        if cached:
            try:
                candidate = np.load(io.BytesIO(cached), allow_pickle=False)
                if candidate.shape != (len(rows), 384) or not np.isfinite(candidate).all() or not np.allclose(np.linalg.norm(candidate, axis=1), 1, atol=1e-5):
                    raise ValueError("Invalid cached embedding rows")
                matrix = candidate
            except (ValueError, OSError) as exc:
                LOG.warning("Recomputing embedding shard: %s", exc)
        if matrix is None:
            current_batch = max(1, batch_size)
            while True:
                try:
                    matrix = normalized_f32(model.encode(rows, batch_size=current_batch, convert_to_numpy=True,
                                                        show_progress_bar=False), 384)
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or current_batch == 1:
                        raise
                    current_batch = max(1, current_batch // 2)
                    LOG.warning("Embedding memory pressure: retry batch_size=%s", current_batch)
            stream = io.BytesIO()
            np.save(stream, matrix, allow_pickle=False)
            store.save_blob("embedding_shards", part_key, stream.getvalue(), {"start": start, "count": len(rows), "config": config})
        parts.append(matrix)
        print(f"Embedding checkpoint: {min(start + len(rows), len(chunks))}/{len(chunks)} rows", flush=True)
    matrix = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    store.save("embedding_manifests", key, {"config": config, "shape": list(matrix.shape), "sha256": digest(matrix.tobytes())})
    return matrix, key


def build_index(matrix, chunks, store, embedding_key):
    import faiss
    import numpy as np
    matrix = normalized_f32(matrix, 384)
    key = digest({"embedding_key": embedding_key, "vectors": digest(matrix.tobytes()), "metric": "IndexFlatIP",
                  "faiss": faiss.__version__, "chunks": [digest(x) for x in chunks]})
    body = store.load_blob("indexes", key)
    index = None
    if body:
        try:
            candidate = faiss.deserialize_index(np.frombuffer(body, dtype=np.uint8).copy())
            if candidate.d != matrix.shape[1] or candidate.ntotal != len(chunks) or candidate.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError("Cached index has incompatible shape or metric")
            index = candidate
        except Exception as exc:
            LOG.warning("Rebuilding invalid FAISS index: %s", exc)
    if index is None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        store.save_blob("indexes", key, faiss.serialize_index(index).tobytes(), {"embedding_key": embedding_key, "d": index.d, "ntotal": index.ntotal})
    return index, key


def retrieve_records(query, model, index, chunks, metadata=None, k=2, kind=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    if index is None or not chunks or index.ntotal == 0:
        return []
    if index.ntotal != len(chunks):
        raise ValueError("FAISS count does not match current chunks")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    expanded = ("thesis submission procedure " if kind == "thesis" else "schedule exam calendar " if kind == "schedule" else "") + query
    vector = normalized_f32(model.encode(expanded, convert_to_numpy=True, show_progress_bar=False), index.d)
    count = min(index.ntotal, max(k, 8) if kind else k)
    distances, ids = index.search(vector, count)
    records = []
    for score, identifier in zip(distances[0], ids[0]):
        number = int(identifier)
        if number < 0 or number >= len(chunks):
            continue
        text = chunks[number]
        lowered = text.lower()
        if kind == "thesis" and "thesis" not in lowered:
            continue
        if kind == "thesis":
            program_patterns = [(r"\bph\.?d\b", r"\bph\.?d\b"), (r"\bm\.?tech\b", r"\bm\.?tech\b"),
                                (r"\bmsr\b", r"\bmsr\b"), (r"\bm\.?des\b", r"\bm\.?des\b")]
            if any(re.search(p, query, re.I) and not re.search(match, text, re.I) for p, match in program_patterns):
                continue
        if kind == "schedule":
            if not any(word in lowered for word in ("exam", "semester", "calendar", "schedule", "registration")):
                continue
            years = re.findall(r"\b20\d{2}\b", query)
            if years and not any(year in text for year in years):
                continue
        info = metadata[number] if metadata and len(metadata) == len(chunks) and metadata[number].get("sha256") == digest(text) else {"sources": []}
        records.append({"chunk_id": number, "score": float(score), "text": text, "sources": info.get("sources", [])})
        if len(records) == k:
            break
    return records


def format_records(records):
    if not records:
        return "No matching evidence was found in the scraped DOAA corpus."
    blocks = []
    for record in records:
        citations = []
        for source in record.get("sources", []):
            label = source["url"] + (f" (page {source['page']})" if source.get("page") else "")
            if label not in citations:
                citations.append(label)
        source_text = "; ".join(citations) or "source metadata unavailable for this supplied chunk"
        blocks.append(f"[C{record['chunk_id']}] cosine={record['score']:.4f}\nSource: {source_text}\n{record['text']}")
    return "\n\n".join(blocks)


def spi_core(grades):
    if not isinstance(grades, str) or not grades.strip() or len(grades) > 20000:
        return "Error: supply comma-separated credits:gradepoint pairs."
    numerator, denominator, count = 0.0, 0.0, 0
    try:
        for pair in grades.split(","):
            fields = pair.split(":")
            if len(fields) != 2 or not all(field.strip() for field in fields):
                raise ValueError("each pair must contain exactly one colon")
            credits, grade = map(float, fields)
            if not math.isfinite(credits) or not math.isfinite(grade) or credits <= 0 or not 0 <= grade <= 10:
                raise ValueError("credits must be positive and finite; grade points must be finite in [0,10]")
            numerator += credits * grade
            denominator += credits
            count += 1
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise ValueError("numeric input is too large")
        result = numerator / denominator
        return f"SPI = {result:.2f} (weighted points {numerator:g} / total credits {denominator:g}; {count} courses)."
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        return f"Error: invalid grades: {exc}. Expected e.g. '4:9,3:8,4:10'."


def parse_judge_response(response):
    if not isinstance(response, str):
        raise ValueError("Judge response is not text")
    match = re.fullmatch(r"\s*SCORE:\s*([1-5])\s*\n\s*REASON:\s*(\S[\s\S]*?)\s*", response, re.I)
    if not match or len(re.findall(r"(?im)^\s*SCORE\s*:", response)) != 1:
        raise ValueError("Expected exactly SCORE: [1-5] followed by a nonempty REASON on the next line")
    return int(match.group(1)), match.group(2).strip()


def parse_json_object(text):
    """Accept one JSON object, optionally fenced, never Python/eval syntax."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def parse_tool_calls(text, schemas):
    """Translate only model-emitted, schema-valid calls; never invent an action."""
    import jsonschema
    allowed = {schema["function"]["name"]: schema["function"]["parameters"] for schema in schemas}
    blocks = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text)
    if "<tool_call>" in text and not blocks:
        raise ValueError("Truncated tool-call XML")
    if not blocks and text.strip().startswith("{"):
        candidate = parse_json_object(text)
        if "name" in candidate or "arguments" in candidate:
            blocks = [stable_json(candidate)]
    calls = []
    for block in blocks:
        obj = parse_json_object(block)
        name, arguments = obj.get("name"), obj.get("arguments")
        if name not in allowed or not isinstance(arguments, dict):
            raise ValueError("Unknown tool or non-object arguments")
        jsonschema.validate(arguments, allowed[name])
        calls.append({"name": name, "args": arguments})
    if len(calls) > 4:
        raise ValueError("Too many tool calls in one model response")
    return calls


def agreement_counts(human_scores, llm_scores, criteria):
    exact = close = total = 0
    for key, row in human_scores.items():
        if any(row.get(criterion, 0) != 0 for criterion in criteria) and not 0 <= int(key) < len(llm_scores):
            raise ValueError("Filled human rating has no corresponding LLM question record")
    for i, row in enumerate(llm_scores):
        human = human_scores.get(i, human_scores.get(str(i), {}))
        for criterion in criteria:
            h = human.get(criterion, 0)
            if h == 0:
                continue
            l = row.get(criterion)
            if type(h) is not int or not 1 <= h <= 5 or type(l) is not int or not 1 <= l <= 5:
                raise ValueError(f"Invalid/missing score at question {i + 1}, criterion {criterion}")
            total += 1
            exact += h == l
            close += abs(h - l) <= 1
    return {"exact": exact, "close": close, "total": total,
            "exact_percent": 100 * exact / total if total else None,
            "close_percent": 100 * close / total if total else None}


def publish_runtime_bundle(root, corpus, chunks, metadata, index, embedding_key, index_key, model_revision):
    import faiss
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": chunks, "metadata": metadata}
    chunk_data = (stable_json(payload) + "\n").encode("utf-8")
    index_data = faiss.serialize_index(index).tobytes()
    corpus_meta = {key: corpus[key] for key in ("status", "corpus_hash", "spans", "visited", "failures", "base_url", "mode", "created_at", "html_parser")}
    corpus_meta["documents"] = [{k: v for k, v in document.items() if k not in {"pages", "links"}} for document in corpus["documents"]]
    corpus_data = (stable_json(corpus_meta) + "\n").encode("utf-8")
    version = digest(chunk_data + index_data)[:20]
    names = {"chunks": f"chunks-{version}.json", "index": f"index-{version}.faiss", "corpus_metadata": f"corpus-{version}.json"}
    atomic_bytes(folder / names["chunks"], chunk_data)
    atomic_bytes(folder / names["index"], index_data)
    atomic_bytes(folder / names["corpus_metadata"], corpus_data)
    manifest = {"schema": 1, "producer": VERSION, "corpus_hash": corpus["corpus_hash"],
                "corpus_status": corpus["status"], "mode": corpus["mode"], "files": names,
                "sha256": {"chunks": digest(chunk_data), "index": digest(index_data), "corpus_metadata": digest(corpus_data)},
                "embedding_model": EMBEDDING_MODEL, "embedding_revision": model_revision,
                "embedding_key": embedding_key, "index_key": index_key, "dimension": index.d,
                "count": index.ntotal, "chunk_size": 2048, "overlap": 0, "created_at": utc_now()}
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def restore_published_corpus(project_root, runtime_root, base_url):
    """Reuse the genuine published text snapshot without shipping raw downloads."""
    folder = Path(runtime_root)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("files", {}).get("corpus_metadata")
    if not name:
        return None  # Earlier bundle format; rebuild from ordinary checkpoints.
    if Path(name).name != name or manifest.get("producer") != VERSION:
        raise ValueError("Invalid published corpus manifest")
    body = (folder / name).read_bytes()
    if digest(body) != manifest["sha256"]["corpus_metadata"]:
        raise ValueError("Published corpus metadata checksum mismatch")
    corpus = json.loads(body)
    text = (Path(project_root) / "RAGretrievalcontentfile.txt").read_text(encoding="utf-8")
    if (digest(text) != corpus["corpus_hash"] or corpus["corpus_hash"] != manifest["corpus_hash"]
            or corpus["base_url"] != normalize_doaa_url(base_url) or corpus["html_parser"] != HTML_PARSER_VERSION
            or corpus["mode"] != "full"):
        raise ValueError("Published text/source/configuration mismatch")
    if any(not 0 <= span["start"] < span["end"] <= len(text) for span in corpus["spans"]):
        raise ValueError("Invalid published source spans")
    corpus["text"] = text
    print("Frozen published corpus restored from checksummed text/provenance; original extraction results reused.")
    return corpus


def load_runtime_bundle(root):
    import faiss
    import numpy as np
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["producer"] != VERSION or manifest["embedding_model"] != EMBEDDING_MODEL:
        raise ValueError("Runtime bundle version/model mismatch; run Q1 again")
    payloads = {}
    for kind in ("chunks", "index"):
        name = manifest["files"][kind]
        if Path(name).name != name:
            raise ValueError("Unsafe runtime-bundle filename")
        data = (root / name).read_bytes()
        if digest(data) != manifest["sha256"][kind]:
            raise ValueError(f"Corrupt runtime bundle {kind}; rerun Q1")
        payloads[kind] = data
    records = json.loads(payloads["chunks"])
    chunks = records["chunks"]
    metadata = records["metadata"]
    index = faiss.deserialize_index(np.frombuffer(payloads["index"], dtype=np.uint8).copy())
    if (index.d != 384 or index.ntotal != len(chunks) or len(metadata) != len(chunks)
            or index.metric_type != faiss.METRIC_INNER_PRODUCT or digest("".join(chunks)) != manifest["corpus_hash"]
            or any(digest(text) != item["sha256"] for text, item in zip(chunks, metadata))):
        raise ValueError("Runtime bundle shape/order/corpus mismatch")
    return chunks, metadata, index, manifest
