"""Durable per-paper results and bounded retries; no third-party dependencies."""
import hashlib
import json
import os
import time
from pathlib import Path

FIELDS = ("tldr", "motivation", "method", "result", "conclusion")


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_with_checkpoint(items, processor, directory, signature, attempts=3, sleep=time.sleep):
    path = Path(directory) / (signature + ".json")
    state = {"success": {}, "pending": {}}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    # Retry unfinished papers even when they have left the daily arXiv list.
    work = dict(state["pending"])
    work.update({item["id"]: item for item in items})
    output = []
    reused = failed = 0
    for index, (paper_id, original) in enumerate(work.items(), 1):
        item = dict(original)
        item.pop("AI", None)
        item.pop("AI_status", None)
        key = hashlib.sha256(json.dumps(
            [paper_id, item.get("summary", "")], ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        cached = state["success"].get(key)
        if cached is not None:
            output.append(cached)
            state["pending"].pop(paper_id, None)
            reused += 1
            continue
        # Persist pending input before spending an API request.
        state["pending"][paper_id] = item
        atomic_write(path, state)
        for attempt in range(attempts):
            try:
                result = processor(dict(item))
                if result is not None:
                    ai = result.get("AI", {})
                    if not all(isinstance(ai.get(k), str) and ai[k].strip() for k in FIELDS):
                        raise ValueError("Missing or empty AI fields")
                break
            except Exception as error:
                print(f"Paper {paper_id}: attempt {attempt + 1}/{attempts} failed ({type(error).__name__})",
                      flush=True)
                if attempt + 1 == attempts:
                    result = dict(item)
                    result["AI"] = {field: "AI summary unavailable; retry pending." for field in FIELDS}
                    result["AI_status"] = "failed"
                    failed += 1
                else:
                    sleep(min(2 ** (attempt + 1), 30))
        if result is None:
            # Positively filtered content is not published.
            state["pending"].pop(paper_id, None)
        else:
            output.append(result)
            if result.get("AI_status") != "failed":
                result["AI_status"] = "success"
                state["success"][key] = result
                state["pending"].pop(paper_id, None)
        atomic_write(path, state)
        print(f"Processed {index}/{len(work)}; reused={reused}, failed={failed}", flush=True)
    atomic_write(path, state)
    print(f"AI batch: {len(work)} papers, {reused} reused, {failed} pending retry", flush=True)
    return output

