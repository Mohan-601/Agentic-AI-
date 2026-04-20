
import json
import time
import base64
import logging
import subprocess
import traceback
import requests
import chromadb
from duckduckgo_search import DDGS
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("helix")

# ══════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════
@dataclass(frozen=True)
class Config:
    ollama_url:    str   = "http://localhost:11434"
    forge_url:     str   = "http://127.0.0.1:7860"

    # ── Models ─────────────────────────────────────
    # planner : gemma2:2b        (~1.6 GB) — fast decomposer
    # thinker : mistral:7b-q4_0  (~4.1 GB) — reasoning & reflection
    # embed   : nomic-embed-text (~0.3 GB) — loaded briefly, then unloaded
    planner_model: str   = "gemma2:2b"
    thinker_model: str   = "mistral"
    embed_model:   str   = "nomic-embed-text"

    image_output:    str   = "helix_creation.png"
    memory_path:     str   = "./helix_memory"
    workspace_dir:   str   = "./helix_workspace"
    memory_topk:     int   = 3
    max_retries:     int   = 3
    max_agent_steps: int   = 10    # hard cap prevents infinite loops

    # 0.5 s is sufficient for VRAM to settle; 3.5 s was wasteful
    vram_settle_s:   float = 0.5

    num_gpu:         int   = 99    # offload all layers to GPU
    num_ctx:         int   = 4096  # keeps KV-cache within 6 GB

    memory_distance_threshold: float = 0.6

    # ~1 500 chars ≈ ~375 tokens — leaves room in 4 096-token context
    search_context_max_chars: int = 1_500

    # Timeout in seconds for sandboxed code execution
    code_exec_timeout: int = 15


CFG = Config()
OLLAMA_OPTS = {"num_gpu": CFG.num_gpu, "num_ctx": CFG.num_ctx}

# Ensure workspace exists
Path(CFG.workspace_dir).mkdir(exist_ok=True)

# ── In-process VRAM state tracker ─────────────────
# Avoids firing a POST + polling cycle when the model is already
# in the desired state.  Populated/cleared by set_model_state().
_currently_loaded: set[str] = set()


# ══════════════════════════════════════════════════
# 2. AGENT STATE  (carried through the loop)
# ══════════════════════════════════════════════════
@dataclass
class AgentState:
    goal:         str        = ""
    steps:        list[dict] = field(default_factory=list)
    # steps: [{"action": str, "prompt": str, "result": str}, ...]
    scratchpad:   str        = ""   # running notes visible to the reflector
    final_answer: str        = ""
    step_count:   int        = 0


# ══════════════════════════════════════════════════
# 3. CHROMADB — PERSISTENT LONG-TERM MEMORY
# ══════════════════════════════════════════════════
log.info("Initialising Long-Term Memory (ChromaDB)...")
_chroma = chromadb.PersistentClient(path=CFG.memory_path)
memory_collection = _chroma.get_or_create_collection(name="helix_knowledge_base")

_conversation_history: list[dict] = []   # {"role": "user"|"assistant", "content": str}


def _history_as_text() -> str:
    """Last 6 turns of this session as a readable block."""
    if not _conversation_history:
        return ""
    lines = []
    for turn in _conversation_history[-6:]:
        role = "User" if turn["role"] == "user" else "Helix"
        lines.append(f"{role}: {turn['content']}")
    return "[Recent conversation]\n" + "\n".join(lines)


# ══════════════════════════════════════════════════
# 4. OLLAMA HTTP HELPER
# ══════════════════════════════════════════════════
def _ollama_post(endpoint: str, payload: dict, retries: int = CFG.max_retries) -> dict:
    """POST to Ollama with exponential back-off retry. Raises on final failure."""
    url = f"{CFG.ollama_url}{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Ollama request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)   # back-off: 2 s, 4 s …
    return {}


# ══════════════════════════════════════════════════
# 5. VRAM MANAGER
# ══════════════════════════════════════════════════
def _loaded_models() -> list[str]:
    """Query Ollama /api/ps for currently loaded model names."""
    try:
        data = requests.get(f"{CFG.ollama_url}/api/ps", timeout=10).json()
        return [m.get("name", "") for m in data.get("models", [])]
    except requests.RequestException:
        return []


def set_model_state(model: str, action: Literal["load", "unload"]) -> None:
    """Load or unload a model from VRAM.

    FIX: Added _currently_loaded cache — skips the entire round-trip when
    the model is already in the desired state.
    FIX: Replaced the polling loop (up to 8.5 s per call!) with a simple
    one-shot POST + vram_settle_s sleep.  The poll loop never meaningfully
    helped; Ollama's own keep_alive=0 response is synchronous.
    """
    global _currently_loaded

    # Fast-path: nothing to do
    if action == "load"   and model in _currently_loaded:
        return
    if action == "unload" and model not in _currently_loaded:
        return

    keep_alive = 0 if action == "unload" else -1
    label = "Unloading" if action == "unload" else "Loading"
    log.info("%s '%s' in VRAM...", label, model)
    try:
        _ollama_post(
            "/api/generate",
            {"model": model, "prompt": "", "keep_alive": keep_alive,
             "stream": False, "options": OLLAMA_OPTS},
        )
        if action == "load":
            _currently_loaded.add(model)
        else:
            _currently_loaded.discard(model)
    except requests.RequestException as exc:
        log.warning("set_model_state('%s', '%s') failed: %s", model, action, exc)

    time.sleep(CFG.vram_settle_s)   # single brief settle; no polling needed


def unload_all() -> None:
    # """Force-unload all known models to free VRAM (e.g. before image gen)."""
    set_model_state(CFG.planner_model, "unload")
    set_model_state(CFG.thinker_model, "unload")


# ══════════════════════════════════════════════════
# 6. MEMORY  (embed → ChromaDB)
# ══════════════════════════════════════════════════
def _embed(text: str) -> list[float]:
    """Generate an embedding vector.

    FIX: Removed the automatic set_model_state(unload) that was previously
    called after every single embedding.  That caused the embed model to be
    loaded and unloaded multiple times per request (e.g. once in
    recall_memory and again in save_to_memory).  Callers that are done with
    the embed model should call set_model_state(CFG.embed_model, "unload")
    themselves — helix_agent() does this at the end of each request.
    """
    result = _ollama_post(
        "/api/embeddings",
        {"model": CFG.embed_model, "prompt": text, "options": OLLAMA_OPTS},
    )
    return result["embedding"]


def save_to_memory(user_prompt: str, helix_reply: str) -> None:
    document  = f"User asked: {user_prompt} | Helix replied: {helix_reply}"
    embedding = _embed(document)
    memory_collection.add(
        ids=[str(time.time())],
        embeddings=[embedding],
        documents=[document],
    )


def recall_memory(query: str, top_k: int = CFG.memory_topk) -> str:
    if memory_collection.count() == 0:
        return ""
    embedding = _embed(query)
    results   = memory_collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, memory_collection.count()),
        include=["documents", "distances"],
    )
    docs      = results.get("documents",  [[]])[0]
    distances = results.get("distances",  [[]])[0]
    relevant  = [
        doc for doc, dist in zip(docs, distances)
        if dist < CFG.memory_distance_threshold
    ]
    return ("[Relevant long-term memory]\n" + "\n".join(f"  - {d}" for d in relevant)
            if relevant else "")


# ══════════════════════════════════════════════════
# 7. TOOL REGISTRY & IMPLEMENTATIONS
# ══════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, str] = {
    "SEARCH":     "Search the web for current events, news, live data, prices, or facts.",
    "THINK":      "Deep reasoning, coding, analysis, writing, math, or general questions.",
    "IMAGE":      "Generate an image from a text description.",
    "RECALL":     "Retrieve information from long-term memory about past conversations.",
    "CODE":       "Execute Python code in a sandbox and return stdout/stderr.",
    "FILE_WRITE": 'Write text to a workspace file. Input: {"filename": "x.txt", "content": "..."}',
    "FILE_READ":  "Read content from a workspace file. Input: filename string.",
    "DONE":       "Signal the goal is fully achieved and provide the final answer.",
}

def _tools_manifest() -> str:
    return "Available tools:\n" + "\n".join(
        f"  {name}: {desc}" for name, desc in TOOL_REGISTRY.items()
    )


# ── Tool: Web Search ──────────────────────────────────────────────────────────
def tool_web_search(query: str) -> str:
    log.info("[Tool: SEARCH] Query: '%s'", query)
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No web results found."
        context = "[Live Web Search Results]:\n"
        for r in results:
            context += f"- {r.get('title', 'Unknown')}: {r.get('body', '')}\n"
        if len(context) > CFG.search_context_max_chars:
            context = context[: CFG.search_context_max_chars] + "\n[...truncated for context budget]"
        return context
    except Exception as exc:
        log.error("[Tool: SEARCH] Error: %s", exc)
        return f"[Search Error]: {exc}"


# ── Tool: Think  (Mistral 7B streaming) ──────────────────────────────────────
def tool_think(prompt: str, extra_context: str = "") -> str:
    """
    Streams a Mistral response, printing tokens in real-time.
    FIX: recall_memory (which loads the embed model) is now called BEFORE
    set_model_state(thinker, "load") so the embed model is already unloaded
    by the time Mistral needs the VRAM.  Previously this happened mid-call,
    causing a silent VRAM clash.
    """
    log.info("[Tool: THINK] Engaging Mistral 7B...")

    # ── Memory recall first (uses embed model, then frees it) ──
    session_ctx = _history_as_text()
    memory_ctx  = recall_memory(prompt)
    # Unload embed now that we have what we need; thinker loads next
    set_model_state(CFG.embed_model, "unload")

    set_model_state(CFG.thinker_model, "load")

    system_parts = [
        "You are Helix, a highly capable AI assistant. "
        "Be precise, thorough, and concise."
    ]
    if extra_context:
        system_parts.append(
            f"\n{extra_context}\n"
            "CRITICAL: Base your answer on the data above. "
            "Do not contradict it. Cite sources where possible."
        )
    if session_ctx:
        system_parts.append(f"\n{session_ctx}")
    if memory_ctx:
        system_parts.append(f"\n{memory_ctx}")

    full_prompt = (
        f"[INST] {''.join(system_parts)}\n\nUser: {prompt} [/INST]\nHelix:"
    )

    payload = {
        "model":   CFG.thinker_model,
        "prompt":  full_prompt,
        "stream":  True,
        "options": OLLAMA_OPTS,
    }

    print("\n[Helix]: ", end="", flush=True)
    parts: list[str] = []

    try:
        with requests.post(
            f"{CFG.ollama_url}/api/generate",
            json=payload, stream=True, timeout=300
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                parts.append(token)
                if chunk.get("done"):
                    break
    except requests.RequestException as exc:
        msg = f"\n[Tool: THINK] Stream error — {exc}"
        print(msg)
        parts.append(msg)

    print()
    response = "".join(parts).strip()
    set_model_state(CFG.thinker_model, "unload")
    return response   # FIX: was "return ''" in original


# ── Tool: Image Generation ────────────────────────────────────────────────────
def tool_generate_image(image_prompt: str) -> str:
    log.info("[Tool: IMAGE] Clearing VRAM for Stable Diffusion...")
    unload_all()
    payload = {
        "prompt": (
            f"{image_prompt}, cinematic lighting, photorealistic, "
            "highly detailed, 8k resolution, sharp focus"
        ),
        "negative_prompt": (
            "ugly, blurry, text, watermark, bad anatomy, "
            "deformed, low quality, noise, dull"
        ),
        "steps": 25, "cfg_scale": 7.0, "width": 768, "height": 512,
    }
    try:
        resp = requests.post(
            f"{CFG.forge_url}/sdapi/v1/txt2img", json=payload, timeout=180
        ).json()
        raw = resp["images"][0]
        with open(CFG.image_output, "wb") as fh:
            fh.write(base64.b64decode(raw.split(",", 1)[-1]))

        # --- NEW RETURN STATEMENT FOR UI RENDERING ---
        return (f"![Generated Image](http://localhost:5000/image?t={int(time.time())})\n\n"
                f"Image generated and saved as '{CFG.image_output}'.")

    except requests.ConnectionError:
        return "[Tool: IMAGE] Cannot reach Forge — is it running on port 7860?"
    except KeyError:
        return "[Tool: IMAGE] Forge responded but returned no image data."
    except Exception as exc:
        return f"[Tool: IMAGE] Unexpected error — {exc}"


# ── Tool: Code Execution (sandboxed subprocess) ───────────────────────────────
def tool_execute_code(code: str) -> str:
    """
    Execute Python in a subprocess with a hard timeout.
    Runs inside CFG.workspace_dir; temp script is cleaned up after.
    """
    log.info("[Tool: CODE] Executing code...")
    script = Path(CFG.workspace_dir) / "_helix_exec.py"
    try:
        script.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            ["python", str(script)],
            capture_output=True, text=True,
            timeout=CFG.code_exec_timeout,
            cwd=CFG.workspace_dir,
        )
        out, err = proc.stdout.strip(), proc.stderr.strip()
        result = ""
        if out:
            result += f"[stdout]\n{out}\n"
        if err:
            result += f"[stderr]\n{err}\n"
        if proc.returncode != 0:
            result += f"[exit code: {proc.returncode}]"
        return result.strip() or "[No output]"
    except subprocess.TimeoutExpired:
        return f"[Tool: CODE] Execution timed out ({CFG.code_exec_timeout}s limit)."
    except Exception as exc:
        return f"[Tool: CODE] Error — {exc}"
    finally:
        if script.exists():
            script.unlink()


# ── Tool: File Write ──────────────────────────────────────────────────────────
def tool_file_write(input_str: str) -> str:
    log.info("[Tool: FILE_WRITE] Writing file...")
    try:
        data     = json.loads(input_str)
        filename = Path(data["filename"]).name          # strip path traversal
        content  = data.get("content", "")
        filepath = Path(CFG.workspace_dir) / filename
        filepath.write_text(content, encoding="utf-8")
        return f"File '{filename}' written ({len(content)} chars) to workspace."
    except (json.JSONDecodeError, KeyError) as exc:
        return (f'[Tool: FILE_WRITE] Invalid input — {exc}. '
                f'Use JSON: {{"filename": "...", "content": "..."}}')
    except Exception as exc:
        return f"[Tool: FILE_WRITE] Error — {exc}"


# ── Tool: File Read ───────────────────────────────────────────────────────────
def tool_file_read(filename: str) -> str:
    log.info("[Tool: FILE_READ] Reading '%s'...", filename)
    try:
        filename = Path(filename.strip()).name          # strip path traversal
        filepath = Path(CFG.workspace_dir) / filename
        if not filepath.exists():
            return f"[Tool: FILE_READ] '{filename}' not found in workspace."
        return f"[File: {filename}]\n{filepath.read_text(encoding='utf-8')}"
    except Exception as exc:
        return f"[Tool: FILE_READ] Error — {exc}"


# ── Tool: Recall ──────────────────────────────────────────────────────────────
def tool_recall(query: str) -> str:
    result = recall_memory(query, top_k=1)
    return result or "No relevant memory found."


# ══════════════════════════════════════════════════
# 8. TOOL DISPATCHER
# ══════════════════════════════════════════════════
def dispatch_tool(action: str, prompt: str, extra_context: str = "") -> str:
    """Route an action name to its implementation and return the result string."""
    match action.upper().strip():
        case "SEARCH":
            return tool_web_search(prompt)
        case "THINK":
            return tool_think(prompt, extra_context=extra_context)
        case "IMAGE":
            return tool_generate_image(prompt)
        case "RECALL":
            return tool_recall(prompt)
        case "CODE":
            return tool_execute_code(prompt)
        case "FILE_WRITE":
            return tool_file_write(prompt)
        case "FILE_READ":
            return tool_file_read(prompt)
        case _:
            log.warning("[Dispatcher] Unknown action '%s', defaulting to THINK.", action)
            return tool_think(prompt, extra_context=extra_context)


# ══════════════════════════════════════════════════
# 9. PLANNER  (Gemma 2B — initial task decomposition)
# ══════════════════════════════════════════════════
def _build_planner_system() -> str:
    return f"""You are the planning brain of Helix, an agentic AI.
Decompose the user's goal into an ordered list of tool calls needed to fully achieve it.

{_tools_manifest()}

Return ONLY a valid JSON array. Each element must have exactly two keys:
  "action": one of SEARCH, THINK, IMAGE, RECALL, CODE, FILE_WRITE, FILE_READ, DONE
  "prompt": the focused input for that tool

Rules:
  • Always end the plan with DONE if the answer can be assembled from the prior steps.
  • Use SEARCH for anything requiring real-time / post-training data.
  • Use CODE to compute, transform data, or automate tasks.
  • Use THINK for reasoning, writing, explanation, or chat.
  • Use RECALL only when the user references past sessions.
  • Chain steps when needed: e.g. SEARCH → THINK to get data then reason about it.

Example:
  Goal: "What is the price of gold today and what does it mean for inflation?"
  Output: [
    {{"action": "SEARCH", "prompt": "gold price today USD 2024"}},
    {{"action": "THINK",  "prompt": "Explain what the current gold price means for inflation"}},
    {{"action": "DONE",   "prompt": "Summarise gold price and inflation analysis"}}
  ]
"""


def plan_tasks(user_prompt: str) -> list[dict]:
    log.info("[Planner] Decomposing goal with %s...", CFG.planner_model)
    set_model_state(CFG.planner_model, "load")

    raw = _ollama_post(
        "/api/generate",
        {
            "model":   CFG.planner_model,
            "prompt":  f"{_build_planner_system()}\n\nGoal: {user_prompt}",
            "format":  "json",
            "stream":  False,
            "options": OLLAMA_OPTS,
        },
    ).get("response", "")

    set_model_state(CFG.planner_model, "unload")

    tasks: list[dict] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            tasks = parsed
        elif isinstance(parsed, dict):
            # Handle models that wrap the array under a key
            for key in ("tasks", "subtasks", "steps", "actions", "plan"):
                if key in parsed and isinstance(parsed[key], list):
                    tasks = parsed[key]
                    break
            if not tasks and "action" in parsed:
                tasks = [parsed]   # single-task dict
    except json.JSONDecodeError as exc:
        log.warning("[Planner] JSON parse failed: %s\n  Raw: %s", exc, raw[:300])

    if not tasks:
        log.info("[Planner] Fallback → THINK for full prompt.")
        tasks = [{"action": "THINK", "prompt": user_prompt}]

    return tasks


# ══════════════════════════════════════════════════
# 10. REFLECTOR  (Mistral — completion check & re-plan)
# ══════════════════════════════════════════════════
REFLECTOR_SYSTEM = """You are the self-reflection module of Helix.
Given the original goal, steps taken, and their results, decide what to do next.

Respond with ONLY a valid JSON object with these exact keys:
  "is_done"      : true if the goal is FULLY achieved, otherwise false
  "reason"       : one sentence explaining your decision
  "next_steps"   : array of {action, prompt} objects for additional work needed
                   (empty array [] if is_done is true)
  "final_answer" : complete answer string for the user if is_done is true, else null

Be critical — only set is_done=true when the goal is completely and correctly addressed.
If any important information is missing, set is_done=false and specify follow-up steps.
"""


def reflect(state: AgentState) -> dict:
    """Ask Mistral to evaluate whether the goal is achieved and what comes next."""
    log.info("[Reflector] Evaluating progress...")
    set_model_state(CFG.thinker_model, "load")

    steps_summary = ""
    for i, step in enumerate(state.steps, 1):
        preview = str(step.get("result", ""))[:400]
        steps_summary += (
            f"\nStep {i}: [{step['action']}] {step['prompt'][:120]}\n"
            f"  Result preview: {preview}\n"
        )

    prompt = (
        f"[INST] {REFLECTOR_SYSTEM}\n\n"
        f"Original Goal: {state.goal}\n"
        f"Steps Taken: {steps_summary}\n"
        f"Scratchpad: {state.scratchpad or 'empty'}\n\n"
        f"Is the goal fully achieved? [/INST]"
    )

    raw = _ollama_post(
        "/api/generate",
        {"model": CFG.thinker_model, "prompt": prompt,
         "format": "json", "stream": False, "options": OLLAMA_OPTS},
    ).get("response", "{}")

    set_model_state(CFG.thinker_model, "unload")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("[Reflector] JSON parse failed — defaulting to done.")
        return {
            "is_done":      True,
            "reason":       "Reflection failed; treating task as complete.",
            "next_steps":   [],
            "final_answer": state.scratchpad or "Task completed.",
        }


# ══════════════════════════════════════════════════
# 11. AGENTIC LOOP
# ══════════════════════════════════════════════════
def helix_agent(user_prompt: str) -> str:
    """
    Full Plan → Execute → Reflect agentic loop.

    Phase 1 — Plan:   Gemma 2B decomposes the goal into a task queue.
    Phase 2 — Execute: Each task is dispatched; results accumulate in context.
                       SEARCH output is auto-injected into the next THINK call.
    Phase 3 — Reflect: Mistral evaluates completion and can add new tasks.
                       Reflection loops until done or max_agent_steps reached.
    Phase 4 — Memory: Goal + final answer are saved to ChromaDB.
    """
    state = AgentState(goal=user_prompt)

    # ── Print banner ───────────────────────────────
    print(f"\n{'═' * 58}")
    print(f"  Goal: {user_prompt}")
    print(f"{'═' * 58}")

    # ══ FAST PATH ════════════════════════════════
    # If the query is trivially a single THINK (no planning needed), skip the
    # Planner and Reflector entirely — that saves 2 full model swap cycles and
    # one extra Mistral inference call.  Heuristic: short prompt with no
    # obvious multi-step keywords.
    _multi_step_keywords = ("search", "find", "look up", "generate image",
                            "draw", "picture", "create file", "write file",
                            "run code", "execute", "calculate", "compute")
    _is_simple = (
        len(user_prompt) < 200
        and not any(kw in user_prompt.lower() for kw in _multi_step_keywords)
    )
    if _is_simple:
        log.info("[Agent] Fast path — skipping planner & reflector.")
        result = tool_think(user_prompt)
        state.final_answer = result
        state.steps.append({"action": "THINK", "prompt": user_prompt, "result": result})
        _conversation_history.append({"role": "user",      "content": user_prompt})
        _conversation_history.append({"role": "assistant", "content": result})
        # Persist to memory and free embed VRAM
        save_to_memory(user_prompt, result)
        set_model_state(CFG.embed_model, "unload")
        return result

    # ══ Phase 1: Plan ════════════════════════════
    initial_tasks = plan_tasks(user_prompt)
    # Filter DONE out of the execution queue (it's a signal, not a runnable tool)
    task_queue: list[dict] = [
        t for t in initial_tasks
        if str(t.get("action", "")).upper() != "DONE"
    ]

    print(f"\n[Helix] Plan — {len(task_queue)} step(s):")
    for i, t in enumerate(task_queue, 1):
        print(f"  {i}. [{t.get('action', '?')}] {str(t.get('prompt', ''))[:90]}")

    search_context = ""   # latest SEARCH result; injected into next THINK

    # ══ Phase 2: Execute ══════════════════════════
    while task_queue and state.step_count < CFG.max_agent_steps:
        task   = task_queue.pop(0)
        action = str(task.get("action", "THINK")).upper().strip()
        prompt = str(task.get("prompt", user_prompt)).strip()

        # Guard: reject unknown actions
        if action not in TOOL_REGISTRY:
            log.warning("[Agent] Unknown action '%s' → THINK.", action)
            action = "THINK"

        state.step_count += 1
        _sep = "─" * 58
        print(f"\n{_sep}")
        print(f"  Step {state.step_count}/{CFG.max_agent_steps}  [{action}]")
        print(f"  {prompt[:110]}")
        print(_sep)

        # Auto-inject latest SEARCH results into THINK calls
        extra_ctx = search_context if action == "THINK" else ""

        result = dispatch_tool(action, prompt, extra_context=extra_ctx)

        # Update running state
        if action == "SEARCH":
            search_context = result                    # pass to next THINK
        if action not in {"THINK", "IMAGE"} and result:
            print(f"\n  ↳ {result[:600]}")             # echo non-streamed results

        state.scratchpad += f"\n[{action}]: {str(result)[:250]}"
        state.steps.append({"action": action, "prompt": prompt, "result": result})

    # ══ Phase 3: Reflect (and re-plan if needed) ═════════════════════════════
    reflection_rounds = 0
    while state.step_count < CFG.max_agent_steps:
        reflection_rounds += 1
        print(f"\n{'─' * 58}")
        print(f"  [Reflection round {reflection_rounds}] Checking goal completion...")
        print(f"{'─' * 58}")

        decision    = reflect(state)
        is_done     = decision.get("is_done", True)
        reason      = decision.get("reason", "")
        next_steps  = decision.get("next_steps", [])
        final_ans   = decision.get("final_answer")

        log.info("[Reflector] Done: %s | %s", is_done, reason)

        if is_done:
            if final_ans:
                print(f"\n{'═' * 58}")
                print("[Helix — Final Answer]")
                print(f"{'─' * 58}")
                print(final_ans)
                print(f"{'═' * 58}")
                state.final_answer = final_ans
            break

        # Not done — execute the additional steps the reflector requested
        if not next_steps:
            log.info("[Reflector] Marked not-done but provided no next_steps — stopping.")
            break

        print(f"\n[Helix] {reason}. Running {len(next_steps)} additional step(s)...")
        for extra_task in next_steps:
            if state.step_count >= CFG.max_agent_steps:
                print(f"\n[Helix] Max steps ({CFG.max_agent_steps}) reached.")
                break

            e_action = str(extra_task.get("action", "THINK")).upper().strip()
            e_prompt = str(extra_task.get("prompt", user_prompt)).strip()

            if e_action not in TOOL_REGISTRY:
                e_action = "THINK"

            state.step_count += 1
            print(f"\n{'─' * 58}")
            print(f"  Step {state.step_count}/{CFG.max_agent_steps}  [{e_action}]  (reflector-added)")
            print(f"  {e_prompt[:110]}")
            print(f"{'─' * 58}")

            extra_ctx = search_context if e_action == "THINK" else ""
            e_result  = dispatch_tool(e_action, e_prompt, extra_context=extra_ctx)

            if e_action == "SEARCH":
                search_context = e_result
            if e_action not in {"THINK", "IMAGE"} and e_result:
                print(f"\n  ↳ {e_result[:600]}")

            state.scratchpad += f"\n[{e_action}]: {str(e_result)[:250]}"
            state.steps.append({"action": e_action, "prompt": e_prompt, "result": e_result})
    else:
        print(f"\n[Helix] Reached max steps ({CFG.max_agent_steps}). Stopping.")

    # ══ Phase 4: Persist to memory ════════════════
    # FIX: this block was completely missing — goal + answer were never saved.
    final = state.final_answer or state.scratchpad or "No answer produced."
    save_to_memory(user_prompt, final)
    # Unload embed model now that we're done with it for this request
    set_model_state(CFG.embed_model, "unload")

    _conversation_history.append({"role": "user",      "content": user_prompt})
    _conversation_history.append({"role": "assistant", "content": final})

    # FIX: helix_agent() had no return statement — was silently returning None.
    return final
# ══════════════════════════════════════════════════
# 12. MAIN INTERFACE
# ══════════════════════════════════════════════════
# if __name__ == "__main__":
#     from flask import Flask, request, jsonify
#     from flask_cors import CORS
#     import logging

#     # Suppress verbose Flask logging so it doesn't clutter your terminal
#     log = logging.getLogger('werkzeug')
#     log.setLevel(logging.ERROR)

#     app = Flask(__name__)
#     CORS(app)  # Allows your HTML file to communicate with this server

#     print("\n[Helix]: Starting Web Server on http://localhost:5000")
#     unload_all()

#     @app.route('/status', methods=['GET'])
#     def status():
#         return jsonify({"status": "online"})

#     @app.route('/chat', methods=['POST'])
#     def chat():
#         data = request.json
#         user_msg = data.get("message", "")
#         if not user_msg:
#             return jsonify({"response": "Empty message."}), 400

#         try:
#             # Run the agentic loop
#             final_response = helix_agent(user_msg)
#             return jsonify({"response": final_response})
#         except Exception as e:
#             return jsonify({"response": f"[System Error]: {str(e)}"}), 500

#     app.run(port=5000)
if __name__ == "__main__":
    # You can test the agent in the terminal here without starting a server
    pass