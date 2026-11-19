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
RETRIEVAL_VERSION = "a6-retrieval-5-primary-source-coverage"
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


def embedding_configuration(chunks, model_key):
    return {"model": EMBEDDING_MODEL, "model_key": model_key, "chunks": [digest(x) for x in chunks],
            "normalized": True, "dtype": "float32", "version": VERSION}


def encode_batches(chunks, model, store, batch_size=16, shard_size=128, model_key=None):
    import numpy as np
    if not chunks:
        raise ValueError("Cannot embed an empty corpus")
    config = embedding_configuration(chunks, model_key)
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


def requested_calendar_period(query):
    """Map a stated calendar year/semester to an academic period, without guessing a date."""
    years = re.findall(r"\b20\d{2}\b", query)
    if not years:
        return None
    year = int(years[0])
    semester = "I" if re.search(r"\bodd\b|\bfirst\s+semester\b", query, re.I) else "II" if re.search(r"\beven\b|\bsecond\s+semester\b", query, re.I) else None
    academic_range = re.search(r"\b(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})\b", query)
    start = int(academic_range[1]) if academic_range else year - (semester == "II")
    if academic_range and semester == "II":
        year = start + 1
    period = f"{start}-{(start + 1) % 100:02d}-{semester}" if semester else None
    return {"year": year, "semester": semester, "academic_period": period}


def schedule_candidates(query, chunks, metadata):
    """Prefer the actual calendar for the requested year and verify its semester header."""
    requested = requested_calendar_period(query)
    if not requested or not metadata:
        return None, {}, requested
    by_url = {}
    for number, item in enumerate(metadata):
        for source in item.get("sources", []):
            url = source.get("url", "")
            if re.search(rf"/Calendar[-_ ]{requested['year']}\.pdf(?:$|\?)", url, re.I):
                by_url.setdefault(url, set()).add(number)
    eligible, headers = set(), {}
    for url, identifiers in by_url.items():
        source_text = "\n".join(chunks[i] for i in sorted(identifiers))
        period = requested["academic_period"]
        if period:
            start, end, semester = period.split("-")
            pattern = rf"{start}\s*[-–/]\s*(?:20)?{end}\s*[-–/]\s*{semester}(?!I|\d)"
            if not re.search(pattern, source_text, re.I):
                continue
        header_match = re.search(r"EVENTS\s*/\s*ACTIVITIES[^\n]*", source_text, re.I)
        header = header_match[0] if header_match else "Calendar year verified from its source filename; semester header verified in source text."
        for number in identifiers:
            eligible.add(number)
            headers[number] = header
    # With a stated year/semester, an unrelated historical schedule is never a fallback.
    return eligible, headers, requested


def temporal_grounding_issues(question, answer, context):
    requested = requested_calendar_period(question)
    if not requested:
        return []
    issues = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", answer):
        years = {int(y) for y in re.findall(r"\b20\d{2}\b", sentence)}
        wrong = years - {requested["year"]}
        contextual = re.search(r"historical|previous|prior|old\s+calendar|does\s+not\s+(?:establish|verify)|not\s+(?:the|a|an|evidence)|cannot\s+(?:establish|verify)|insufficient", sentence, re.I)
        if wrong and not contextual and re.search(r"exam|falls|scheduled|held|deadline|date|on\s+|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", sentence, re.I):
            issues.append(f"An asserted date/year {sorted(wrong)} contradicts requested calendar year {requested['year']}")
    if context.startswith("No matching evidence") and re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\b\d{4}-\d{2}-\d{2}", answer, re.I):
        issues.append("A specific date was asserted without matching calendar evidence")
    return issues


def generation_truncation_issues(answer, record, max_tokens):
    """Use actual token-budget evidence plus conservative structural signals."""
    if not record or record.get("output_tokens", 0) < max_tokens:
        return []
    text = answer.rstrip()
    issues = []
    if record.get("finish_reason") == "length":
        issues.append("Generation stopped at its token limit")
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if text.count(opening) > text.count(closing):
            issues.append("Generation budget exhausted with an unclosed structure")
    if re.search(r"\b(?:a|an|the|their|its|of|to|with|and|or|including|such as)\s*[*_`]*$", text, re.I):
        issues.append("Generation budget exhausted mid-phrase")
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*$", text):
        issues.append("Generation budget exhausted at an empty list item")
    # Citations, parentheses and numeric list values are legitimate endings.
    complete_ending = re.search(r"[.!?\])`][\s*_'\"`]*$|\b\d+(?:\.\d+)?\s*$", text)
    if not issues and not complete_ending:
        issues.append("Generation exhausted its budget without a complete closing phrase")
    return list(dict.fromkeys(issues))


def verified_calendar_event(query, records):
    """Read an unambiguous event row/semester column from the retrieved real calendar."""
    requested = requested_calendar_period(query)
    if not requested or not requested.get("academic_period"):
        return None
    event = "Mid Semester Examination" if re.search(r"mid[\s-]*(?:sem|term)", query, re.I) else "End Semester Examination" if re.search(r"end[\s-]*(?:sem|term)", query, re.I) else None
    if not event:
        return None
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    date_range = re.compile(rf"\b{month}\.?\s*\d{{1,2}}(?:\s*[-–]\s*(?:{month}\.?\s*)?\d{{1,2}})?\s*,?\s*20\d{{2}}\b", re.I)
    event_pattern = r"(?:^|\n)\s*Mid[\s-]*Semester\s+Examination" if event.startswith("Mid") else r"(?:^|\n)\s*End[\s-]*Semester\s+Examination"
    for record in records:
        header = record.get("calendar_header", "")
        columns = re.findall(r"SEMESTER\s+(20\d{2}\s*[-–/]\s*\d{2}\s*[-–/]\s*(?:II|I))|SUMMER\s+TERM\s+(20\d{2})", header, re.I)
        labels = [re.sub(r"\s+", "", academic).replace("–", "-").replace("/", "-").upper() if academic else "SUMMER " + summer for academic, summer in columns]
        if requested["academic_period"] not in labels:
            continue
        match = re.search(event_pattern, record["text"], re.I)
        if not match:
            continue
        after = record["text"][match.end():]
        boundary = re.search(r"\n\s*(?:Make[\s-]*up\s+Examination|Last date|End Semester Examination|Vacation)", after, re.I)
        row_text = after[:boundary.start()] if boundary else after
        dates = list(date_range.finditer(row_text))
        if len(dates) != len(labels):
            continue  # Never guess alignment when a PDF table has lost a column.
        selected = dates[labels.index(requested["academic_period"])][0]
        if not selected.endswith(str(requested["year"])):
            continue
        return {"event": event, "period": requested["academic_period"],
                "date_range": re.sub(r"\s+", " ", selected), "chunk_id": record["chunk_id"],
                "sources": record.get("sources", []), "source_header": header}
    return None


def format_schedule_records(query, records):
    context = format_records(records)
    event = verified_calendar_event(query, records)
    if event:
        line = (f"Verified calendar event: {event['event']} for {event['period']}: "
                f"{event['date_range']} [C{event['chunk_id']}].")
        return line + "\n\n" + context
    if requested_calendar_period(query) and records:
        return "The requested calendar period was found, but its exact event-date column could not be verified. Do not assert an exam date.\n\n" + context
    return context


def evidence_sentence_catalog(records):
    """Complete source sentences only; every selectable claim retains its real chunk ID."""
    catalog, seen = [], set()
    for record in records:
        marked = re.sub(r"(?m)^\s*(?:[•✔]|[a-z][.)])\s+", "\n\n", record["text"])
        for paragraph in re.split(r"\n\s*\n", marked):
            flat = re.sub(r"\s+", " ", paragraph).strip()
            units = re.split(r"(?<=[.!?])\s+(?=[A-Z•✔]|\d+(?:\.\d+)*\s)", flat)
            for position, unit in enumerate(units):
                if position and re.match(r"^(?:However|This|These|Such|It)\b", unit):
                    unit = units[position - 1] + " " + unit
                unit = re.sub(r"^\d+(?:\.\d+)*\s+", "", unit.strip(" •✔"))
                identity = re.sub(r"[^a-z0-9]+", "", unit.casefold())
                has_clause = re.search(r"\b(?:is|are|was|were|has|have|must|may|shall|should|will|can|cannot|requires?|follows?|reviews?|approves?|submits?|needs?|awards?|offers?|includes?|ends?|commences?|starts?|takes?|provides?|undergo|applies|apply|depends|consists|receives?)\b|=", unit, re.I)
                if (len(unit) < 24 or len(unit) > 900 or not unit[0].isupper()
                        or not unit.endswith((".", "!", "?", ";", ":")) or not has_clause or identity in seen):
                    continue
                assert re.sub(r"\s+", " ", unit) in re.sub(r"\s+", " ", record["text"])
                seen.add(identity)
                occurrence = re.search(r"\s+".join(re.escape(word) for word in unit.split()), record["text"])
                overlaps = []
                if occurrence:
                    overlaps = [(max(0, min(occurrence.end(), source.get("chunk_end", len(record["text"])))
                                     - max(occurrence.start(), source.get("chunk_start", 0))), source.get("url", ""))
                                for source in record.get("sources", [])]
                highest = max((amount for amount, _ in overlaps), default=0)
                urls = sorted({url for amount, url in overlaps if amount == highest and amount > 0})
                catalog.append({"id": len(catalog) + 1, "chunk_id": record["chunk_id"], "text": unit, "source_urls": urls})
    def normalized(value):
        return re.sub(r"[^a-z0-9]+", "", value.casefold())
    catalog = [row for row in catalog if not any(
        row["chunk_id"] == other["chunk_id"] and row["id"] != other["id"]
        and len(row["text"]) < len(other["text"])
        and normalized(row["text"]) in normalized(other["text"])
        for other in catalog)]
    return [{**row, "id": number + 1} for number, row in enumerate(catalog)]


def prioritize_evidence_sentences(question, catalog, limit=10):
    """Bound the selectable evidence inside the same chunks, retaining source/order identity."""
    stop = {"what", "when", "where", "which", "how", "is", "are", "the", "a", "an", "for", "of", "at", "in", "and", "or", "my", "do", "does", "i", "iit", "iitk", "kanpur", "after", "to", "with", "rules"}
    def words(value):
        return {word[:4] if len(word) > 4 else word for word in re.findall(r"[a-z]+", value.lower().replace(".", "")) if word not in stop}
    terms = words(question)
    document_scores = {}
    for row in catalog:
        for url in row.get("source_urls", []):
            document_scores[url] = len(terms & words(urlsplit(url).path.rsplit("/", 1)[-1]))
    best = max(document_scores.values(), default=0)
    pool = [row for row in catalog if best < 2 or any(document_scores.get(url, 0) >= best for url in row.get("source_urls", []))]
    if not pool:
        pool = catalog
    primary_chunk = min(pool, key=lambda row: row["id"])["chunk_id"] if pool else None
    def score(row):
        action = bool(re.search(r"\bmust\b|\bmay\b|\bshall\b|\bonly\b|\bnot\b|\bwill\b|\bcan\b|cannot|exception|requir|approv|submit|review|sign|attach|undergo|subject to", row["text"], re.I))
        return 5 * action + min(3, len(terms & words(row["text"]))) + 2 * (row["chunk_id"] == primary_chunk)
    chosen = sorted(sorted(pool, key=lambda row: (-score(row), row["id"]))[:limit], key=lambda row: row["id"])
    return [{**row, "original_id": row["id"], "id": number + 1} for number, row in enumerate(chosen)]


def parse_sentence_selection(raw, catalog, limit=10):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    selected = json.loads(text)
    allowed = {row["id"] for row in catalog}
    if (not isinstance(selected, list) or len(selected) > limit
            or any(type(number) is not int or number not in allowed for number in selected)
            or len(selected) != len(set(selected))):
        raise ValueError("Select at most ten distinct existing sentence IDs as one JSON array")
    return selected


def evidence_scope_limit(question, records):
    text = "\n".join(record["text"] for record in records)
    urls = {s.get("url", "") for record in records for s in record.get("sources", [])}
    if (re.search(r"\bCPI\b", question, re.I) and re.search(r"calculat|comput|formula", question, re.I)
            and not re.search(r"percentage|conversion|fee|charge|cost", question, re.I)
            and re.search(r"conversion certificate|processing fee|process ing fee", text, re.I)
            and not re.search(r"grade.?points?|weighted|summation|credit.?weight|cumulative.*performance", text, re.I)):
        return "These excerpts concern a conversion certificate and its processing charges; they do not establish the requested academic calculation formula."
    minutes_only = bool(urls) and all(re.search(r"/SPGC[-_]\d", url, re.I) for url in urls)
    if (minutes_only and re.search(r"deadline|when|what is", question, re.I)
            and re.search(r"extension|request", text, re.I)
            and not re.search(r"all students (?:must|shall)|within \w+ months of admission", text, re.I)):
        return "These excerpts record individual historical requests and extensions; they do not establish the requested general definition or deadline."
    return None


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
    eligible, calendar_headers, calendar_request = schedule_candidates(query, chunks, metadata) if kind == "schedule" else (None, {}, None)
    count = index.ntotal if eligible is not None else min(index.ntotal, max(k, 8) if kind else k)
    distances, ids = index.search(vector, count)
    ranked = list(zip(distances[0], ids[0]))
    if eligible is not None:
        event_pattern = r"mid[\s-]*semester\s*(?:examination|exam)" if re.search(r"mid[\s-]*(?:sem|term)", query, re.I) else r"examination|registration|classes"
        ranked = [(score, identifier) for score, identifier in ranked if int(identifier) in eligible]
        ranked.sort(key=lambda pair: (0 if re.search(event_pattern, chunks[int(pair[1])], re.I) else 1 if re.search(r"EVENTS\s*/\s*ACTIVITIES", chunks[int(pair[1])], re.I) else 2, -float(pair[0])))
    records = []
    for score, identifier in ranked:
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
        record = {"chunk_id": number, "score": float(score), "text": text, "sources": info.get("sources", [])}
        if number in calendar_headers:
            record["calendar_request"] = calendar_request
            record["calendar_header"] = calendar_headers[number]
        records.append(record)
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
        calendar_note = ""
        if record.get("calendar_request"):
            period = record["calendar_request"].get("academic_period") or record["calendar_request"]["year"]
            calendar_note = (f"Calendar metadata: requested period {period}. Read only its matching event-date column. "
                             f"Actual source column header: {record['calendar_header']}\n")
        # Similarity is retrieval metadata, never an academic quantity in model evidence.
        blocks.append(f"[C{record['chunk_id']}]\nSource: {source_text}\n{calendar_note}{record['text']}")
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
    # Only cosmetic Markdown around the two labels is normalized. No score is inferred.
    cleaned = re.sub(r"(?im)^\s*(?:\*\*|__)(SCORE|REASON)\s*:(?:\*\*|__)\s*", r"\1: ", response)
    cleaned = re.sub(r"(?im)^\s*(?:\*\*|__)(SCORE|REASON)(?:\*\*|__)\s*:\s*", r"\1: ", cleaned)
    match = re.fullmatch(r"\s*SCORE:\s*([1-5])\s*\n\s*REASON:\s*(\S[\s\S]*?)\s*", cleaned, re.I)
    if not match or len(re.findall(r"(?im)^\s*SCORE\s*:", cleaned)) != 1:
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


def validate_human_review(supplied, evaluation_input_hash, question_count, criteria=CRITERIA):
    """Validate reviewer-supplied ratings bound to the exact evaluation inputs."""
    if supplied.get("evaluation_input_hash") != evaluation_input_hash:
        raise ValueError("Reviewer ratings refer to different answers/context/rubric")
    evaluator = supplied.get("evaluator")
    if not isinstance(evaluator, str) or not evaluator.strip():
        raise ValueError("Reviewer must name their evaluator")
    method = supplied.get("review_method", "independent_human" if supplied.get("independent_human_review") is True else "")
    independent = supplied.get("independent_human_review") is True
    if method == "human_reviewer":
        if independent or supplied.get("reviewer_confirmed") is not True or not isinstance(supplied.get("rating_origin"), str) or not supplied["rating_origin"].strip():
            raise ValueError("Student review requires confirmation and its recorded origin; independent review uses a separate attestation")
    elif method != "independent_human" or not independent:
        raise ValueError("Ratings need reviewer confirmation or an independent-review attestation")
    rows = supplied.get("scores")
    if not isinstance(rows, dict):
        raise ValueError("Reviewer scores must be an object keyed by question number")
    validated = {}
    for key, row in rows.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key or not 0 <= int(key) < question_count:
            raise ValueError("Reviewer rating question ID out of range")
        if not isinstance(row, dict) or set(row) != set(criteria) or any(type(v) is not int or not 0 <= v <= 5 for v in row.values()):
            raise ValueError("Each rating must have three integer scores 1–5, or zero while pending")
        validated[int(key)] = row
    provenance = {"evaluator": evaluator.strip(), "review_method": method, "independent_human_review": independent,
                  "reviewer_confirmed": True, "rating_origin": supplied.get("rating_origin", "Independent reviewer attestation")}
    return validated, provenance


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


def restore_published_embeddings(root, chunks, model_key):
    """Recover already-computed normalized vectors from the packaged exact FAISS index."""
    import numpy as np
    try:
        stored_chunks, _, index, manifest = load_runtime_bundle(root)
        expected_key = digest(embedding_configuration(chunks, model_key))
        if stored_chunks != chunks or manifest["embedding_key"] != expected_key:
            return None
        matrix = np.ascontiguousarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)
        if (matrix.shape != (len(chunks), 384) or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)):
            raise ValueError("Published FAISS vectors are not valid normalized embeddings")
        return {"matrix": matrix, "index": index, "embedding_key": expected_key,
                "index_key": manifest["index_key"], "source": "validated_published_faiss_vectors"}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.warning("Published embeddings unavailable; use validated shards or recompute missing data: %s", exc)
        return None
