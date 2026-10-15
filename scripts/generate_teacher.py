"""Collect resumable WebShop teacher episodes; smoke and full share one run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import uuid

from .common import (DEFAULT_DATASET, DEFAULT_ETO, ETO_COMMIT, ROOT, canonical, observation_message,
                     read_json, read_jsonl, sha256, validate_tasks, write_json, write_jsonl)


@contextmanager
def run_lock(directory):
    """OS releases the advisory lock on process exit, including crashes."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another generator is using this output directory") from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parse_action(text):
    match = re.fullmatch(r"Thought:\s*\S[\s\S]*?\nAction:\s*((?:search|click)\[[^\r\n\[\]]+\])\s*", text.strip())
    if not match or len(re.findall(r"(?m)^Action:", text)) != 1:
        raise ValueError("Expected one Thought and one final search[...] or click[...] action")
    return match.group(1)


def is_full_success(done, reward):
    return bool(done) and math.isfinite(float(reward)) and abs(float(reward) - 1.0) <= 1e-8


def load_prompt(eto_root):
    # ReAct example is context only; never included in the SFT targets.
    return read_json(Path(__file__).parent / "prompts/react_webshop.json")["messages"]


def verify_upstream(eto_root):
    head = subprocess.check_output(["git", "-C", str(eto_root), "rev-parse", "HEAD"], text=True).strip()
    if head != ETO_COMMIT:
        raise ValueError(f"Expected ETO {ETO_COMMIT}, found {head}")
    files = sorted((eto_root / "envs/webshop/src").rglob("*.py"))
    files += sorted((eto_root / "envs/webshop/src").rglob("*.html"))
    files += [eto_root / "eval_agent/prompt/instructions/webshop_inst.txt",
              eto_root / "eval_agent/prompt/icl_examples/webshop_icl.json"]
    hashes = {}
    for path in files:
        relative = path.relative_to(eto_root).as_posix()
        original = subprocess.check_output(["git", "-C", str(eto_root), "show", f"HEAD:{relative}"])
        # Git may check out CRLF on Windows; this is not a semantic source change.
        actual = path.read_bytes().replace(b"\r\n", b"\n")
        if actual != original.replace(b"\r\n", b"\n"):
            raise ValueError(f"Modified upstream file: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return hashes


def check_goal_identity(goals, rows):
    if len(goals) != 12087:
        raise ValueError(f"Full catalog must expose 12087 goals, found {len(goals)}")
    heldout = {canonical(g["instruction_text"]) for g in goals[:1500]}
    for row in rows:
        actual = canonical(goals[row["task_id"]]["instruction_text"])
        if actual != canonical(row["instruction_reference"]):
            raise ValueError(f"Goal identity mismatch for {row['task_id']}; check catalog/order/version")
        if actual in heldout:
            raise ValueError(f"Task {row['task_id']} overlaps held-out instruction text")


def create_environment(eto_root, rows, seed):
    package = eto_root / "envs/webshop"
    assets = [package / "data" / name for name in
              ("items_shuffle.json", "items_ins_v2.json", "items_human_ins.json")]
    index = package / "search_index/indexes"
    if any(not p.is_file() for p in assets) or not index.is_dir():
        raise FileNotFoundError("Full WebShop data/index missing. Follow README.md to install the WebShop data and index first.")
    print("Hashing WebShop data and search index for reproducible resume...", flush=True)
    assets += sorted(p for p in index.rglob("*") if p.is_file() and p.name != "write.lock")
    fingerprint = {p.relative_to(package).as_posix(): sha256(p) for p in assets}
    from .prepare_catalog import prepare_catalog
    catalog, catalog_metadata = prepare_catalog(eto_root, fingerprint["data/items_shuffle.json"])
    fingerprint["catalog_cache"] = catalog_metadata
    sys.path.insert(0, str(package / "src"))
    # Pyjnius first searches javac and can fail on a JRE-only installation.
    # Resolve Ubuntu's /usr/bin/java symlink without changing machine settings.
    if not any(os.getenv(name) for name in ("JAVA_HOME", "JDK_HOME", "JRE_HOME")):
        java = shutil.which("java")
        if java:
            java_home = Path(java).resolve().parent.parent
            if (java_home / "lib/server/libjvm.so").is_file():
                os.environ["JAVA_HOME"] = str(java_home)
    random.seed(seed)  # Product prices and instruction price bounds are sampled during initialization.
    from webshop.web_agent_site.envs import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode="text_rich", human_goals=True, num_products=None, file_path=str(catalog))
    check_goal_identity(env.server.goals, rows)
    # Hash fully materialized goals, including sampled price constraints.
    encoded = json.dumps(env.server.goals, sort_keys=True, ensure_ascii=False).encode()
    fingerprint["materialized_goals_sha256"] = hashlib.sha256(encoded).hexdigest()
    return env, fingerprint


def ensure_manifest(directory, config):
    path = directory / "manifest.json"
    if path.exists():
        old = read_json(path)
        if old != config:
            changed = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            raise ValueError(f"Resume configuration differs: {changed}. Use a new output directory.")
    else:
        if any((directory / "episodes").glob("*.json")):
            raise ValueError("Episode files exist without a manifest")
        write_json(path, config)


def run_episode(env, complete, row, prefix, max_steps, event, context_policy=None):
    from .context_policy import prepare_context
    started = time.monotonic()
    reset_result = env.reset(int(row["task_id"]))  # int is essential: str chooses a random goal upstream.
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    displayed_instruction = re.sub(r"^\s*Instruction:\s*", "", env.instruction_text, count=1)
    if canonical(displayed_instruction) != canonical(row["instruction_reference"]):
        raise ValueError("Reset returned a different task")
    messages = deepcopy(prefix)
    prompt_count = len(messages)
    messages.append({"role": "user", "content": observation_message(observation, env.get_available_actions()["clickables"])})
    event({"kind": "reset", "task_id": row["task_id"], "messages": deepcopy(messages)})
    reward, done, valid, reason = 0.0, False, True, "max_steps"
    steps = []
    for step in range(1, max_steps + 1):
        prompt_messages, context_stats = prepare_context(
            messages, context_policy, prompt_count, task_instruction=displayed_instruction)
        response = complete(prompt_messages)
        text = response["content"]
        messages.append({"role": "assistant", "content": text})
        item = {"step": step, **response}
        if context_stats is not None:
            item["context"] = context_stats
        event({"kind": "response", **item})  # Persist paid output before executing the environment action.
        if response["finish_reason"] != "stop" or not text.strip():
            valid = False
            reason = "incomplete_generation"
            steps.append(item)
            break
        try:
            action = parse_action(text)
        except ValueError:
            valid = False
            item["error"] = "invalid_format"
            observation = "Invalid format. Use Thought: ... followed by one Action: search[...] or Action: click[...]."
        else:
            item["action"] = action
            available = env.get_available_actions()
            name, argument = action.split("[", 1)
            argument = argument[:-1].lower()
            # ETO's SimServer accepts a new search on every page, even where the
            # HTML does not display a search bar. Match its action semantics.
            allowed = ((name == "search" and bool(argument.strip())) or
                       (name == "click" and argument in available["clickables"] and argument != "search"))
            if not allowed:
                valid = False
                item["error"] = "invalid_action"
                observation = "Invalid action!"
            else:
                try:
                    observation, reward, done, _ = env.step(action)
                except AssertionError:
                    valid = False
                    item["error"] = "invalid_action"
                    observation = "Invalid action!"
                else:
                    reward = float(reward)
                    if not math.isfinite(reward) or not 0 <= reward <= 1:
                        raise ValueError(f"Unexpected environment reward: {reward}")
        clickables = [] if done else list(env.get_available_actions()["clickables"])
        item.update(observation=observation, reward=reward, done=bool(done), available_clicks=clickables)
        steps.append(item)
        messages.append({"role": "user", "content": observation_message(observation, clickables, done)})
        event({"kind": "transition", **item})
        if done:
            reason = "purchase"
            break
    success = is_full_success(done, reward)
    # Only clean, complete full-reward trajectories enter the SFT export.
    trainable = success and valid
    # Drop final environment feedback (which can reveal gold goal details).
    # Keep demonstrations separately so their assistant tokens are not trained.
    end = len(messages) - 1 if messages[-1]["role"] == "user" else len(messages)
    sft_messages = [deepcopy(prefix[0]), *deepcopy(messages[prompt_count:end])] if trainable else None
    return {"task_id": row["task_id"], "status": "completed", "reward": reward,
            "success": success, "trainable": trainable, "done": bool(done), "terminate_reason": reason,
            "steps": steps, "messages": messages, "prompt_message_count": prompt_count,
            "sft_messages": sft_messages, "elapsed_seconds": time.monotonic() - started}


def load_episodes(directory, rows):
    episodes = []
    for row in rows:
        path = directory / "episodes" / f"{row['task_id']}.json"
        if path.exists():
            result = read_json(path)
            if result["task_id"] != row["task_id"] or result["status"] != "completed":
                raise ValueError(f"Invalid episode file: {path}")
            episodes.append(result)
    return episodes


def summarize(directory, rows, episodes=None, export=True):
    if episodes is None:
        episodes = load_episodes(directory, rows)
    n = len(episodes)
    successes = sum(r["success"] for r in episodes)
    trainable = [r for r in episodes if r["trainable"]]
    if export:
        write_jsonl(directory / "successful_trajectories.jsonl", trainable)
        write_jsonl(directory / "teacher_sft.jsonl",
                    [{"id": f"webshop_train_{r['task_id']}", "task_id": r["task_id"], "messages": r["sft_messages"]}
                     for r in trainable])
    summary = {"selected_tasks": len(rows), "completed_tasks": n, "remaining_tasks": len(rows) - n,
               "full_reward_successes": successes, "success_rate": successes / n if n else None,
               "mean_reward": sum(r["reward"] for r in episodes) / n if n else None,
               "clean_sft_trajectories": len(trainable),
               "mean_steps": sum(len(r["steps"]) for r in episodes) / n if n else None,
               "completion_tokens": sum(s.get("usage", {}).get("completion_tokens", 0) or 0
                                        for r in episodes for s in r["steps"]),
               "note": "Rates cover completed tasks only. API/interruption attempts remain in attempts/."}
    write_json(directory / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--reuse-run", type=Path, help="Reuse matching lower-step run; replay cached responses for budget-exhausted episodes")
    p.add_argument("--smoke-count", type=int, default=100)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--eto-root", type=Path, default=Path(os.getenv("ETO_ROOT", str(DEFAULT_ETO))))
    p.add_argument("--output-dir", type=Path, default=ROOT / "logs/webshop/qwen35_27b_teacher_clicks20_seed42")
    p.add_argument("--model", default=os.getenv("WEBSHOP_TEACHER_MODEL", "qwen3.5-27b"))
    p.add_argument("--base-url", default=os.getenv("WEBSHOP_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--environment-seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--api-retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="Validate tasks, pinned code and prompt without simulator/API")
    p.add_argument("--check-env", action="store_true", help="Validate full environment/task mapping without API")
    args = p.parse_args()
    if args.smoke_count < 1 or args.max_tokens < 1 or args.max_steps < 1 or args.timeout <= 0 or args.api_retries < 0:
        p.error("Counts and timeout must be positive; retries must be nonnegative")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        p.error("Invalid sampling parameters")
    rows = read_jsonl(args.dataset)
    validate_tasks(rows)
    dataset_manifest = read_json(args.dataset.parent / "manifest.json")
    if dataset_manifest["tasks_sha256"] != sha256(args.dataset):
        raise ValueError("Dataset hash differs from its manifest")
    if len(rows) != 2000:
        raise ValueError("This experiment requires exactly 2000 tasks")
    code_hashes = verify_upstream(args.eto_root)
    prefix = load_prompt(args.eto_root)
    excluded_demo_ids = [row["task_id"] for row in rows
                         if any(canonical(row["instruction_reference"]) in canonical(m["content"])
                                for m in prefix[1:])]
    rows = [row for row in rows if row["task_id"] not in excluded_demo_ids]
    if excluded_demo_ids:
        print(json.dumps({"excluded_prompt_overlap_task_ids": excluded_demo_ids}), flush=True)
    selected = rows[:args.smoke_count] if args.mode == "smoke" else rows
    print(json.dumps({"mode": args.mode, "task_count": len(selected), "dataset_count": len(rows),
                      "first_task_ids": [r["task_id"] for r in selected[:5]], "model": args.model,
                      "max_steps": args.max_steps, "max_tokens": args.max_tokens,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("Dry-run passed. No simulator initialized; no API calls.")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv(args.api_key_env)
    if not args.check_env and not key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
    with run_lock(args.output_dir):
        env, assets = create_environment(args.eto_root, rows, args.environment_seed)
        try:
            if args.check_env:
                print("Environment and all 2000 goal identities verified; no API calls.")
                return
            config = {"schema_version": 1, "dataset_sha256": sha256(args.dataset), "model": args.model,
                      "base_url": args.base_url, "max_tokens": args.max_tokens, "max_steps": args.max_steps,
                      "temperature": args.temperature, "top_p": args.top_p, "enable_thinking": False,
                      "environment_seed": args.environment_seed, "observation_mode": "text_rich", "prompt": prefix,
                      "excluded_prompt_overlap_task_ids": excluded_demo_ids,
                      "eto_commit": ETO_COMMIT, "upstream_code": code_hashes, "assets": assets,
                      "generator_sha256": sha256(Path(__file__)),
                      "common_sha256": sha256(Path(__file__).with_name("common.py"))}
            reuse = {}
            if args.reuse_run:
                source_config = read_json(args.reuse_run / "manifest.json")
                differences = [k for k in set(source_config) | set(config)
                               if k not in {"max_steps", "generator_sha256"} and source_config.get(k) != config.get(k)]
                if differences or source_config["max_steps"] >= args.max_steps:
                    raise ValueError(f"Reuse requires identical settings except increased step budget: {differences}")
                config["reuse_source_manifest_sha256"] = sha256(args.reuse_run / "manifest.json")
                config["reuse_source"] = str(args.reuse_run)
                reuse = {e["task_id"]: e for e in load_episodes(args.reuse_run, selected)}
            ensure_manifest(args.output_dir, config)
            for task_id, cached in reuse.items():
                if cached["terminate_reason"] == "max_steps":
                    continue
                target = args.output_dir / "episodes" / f"{task_id}.json"
                if not target.exists():
                    copied = deepcopy(cached)
                    copied["reused_from"] = str(args.reuse_run)
                    source_attempt = args.reuse_run / cached["attempt_file"]
                    dest_attempt = args.output_dir / cached["attempt_file"]
                    dest_attempt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_attempt, dest_attempt)
                    write_json(target, copied)
            print(json.dumps({"reusable_finished": sum(e["terminate_reason"] != "max_steps" for e in reuse.values()),
                              "cached_prefixes": sum(e["terminate_reason"] == "max_steps" for e in reuse.values())}), flush=True)
            from openai import OpenAI, APIError
            client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout, max_retries=args.api_retries)

            def complete(messages):
                result = client.chat.completions.create(
                    model=args.model, messages=messages, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p, extra_body={"enable_thinking": False})
                choice = result.choices[0]
                return {"content": choice.message.content or "", "finish_reason": choice.finish_reason,
                        "usage": result.usage.model_dump() if result.usage else {},
                        "response_id": result.id, "response_model": result.model}

            try:
                episodes = load_episodes(args.output_dir, rows)
                summarize(args.output_dir, rows, episodes)
                for position, row in enumerate(selected, 1):
                    target = args.output_dir / "episodes" / f"{row['task_id']}.json"
                    if target.exists():
                        continue  # Includes unsuccessful completed episodes; no best-of-N resampling.
                    attempt = args.output_dir / "attempts" / f"{row['task_id']}_{uuid.uuid4().hex}.jsonl"
                    attempt.parent.mkdir(parents=True, exist_ok=True)
                    with attempt.open("a", encoding="utf-8") as log:
                        def event(value):
                            log.write(json.dumps(value, ensure_ascii=False) + "\n")
                            log.flush()
                            os.fsync(log.fileno())
                        try:
                            cached = reuse.get(row["task_id"])
                            cached_index = 0
                            def replay_or_complete(messages):
                                nonlocal cached_index
                                if cached and cached_index < len(cached["steps"]):
                                    if messages != cached["messages"][:len(messages)]:
                                        raise ValueError("Cached replay observation differs; refusing to reuse response")
                                    saved = cached["steps"][cached_index]
                                    cached_index += 1
                                    return {k: saved[k] for k in ("content", "finish_reason", "usage", "response_id", "response_model") if k in saved}
                                if cached and cached_index == len(cached["steps"]):
                                    if messages != cached["messages"]:
                                        raise ValueError("Final cached replay observation differs")
                                    cached_index += 1
                                return complete(messages)
                            episode = run_episode(env, replay_or_complete, row, prefix, args.max_steps, event)
                            if cached:
                                episode["reused_from"] = str(args.reuse_run)
                                episode["reused_response_count"] = len(cached["steps"])
                        except APIError as exc:
                            # SDK retries are exhausted. Preserve paid responses and
                            # leave this task unfinished so a future run can retry it.
                            event({"kind": "error", "type": type(exc).__name__,
                                   "http_status": getattr(exc, "status_code", None),
                                   "disposition": "skip_task"})
                            print(f"[{position}/{len(selected)}] task={row['task_id']} "
                                  f"API failed after retries: {type(exc).__name__}; "
                                  "attempt preserved, skipping to next task.", flush=True)
                            continue
                        except BaseException as exc:
                            # Do not log exception text: SDK errors can contain sensitive request data.
                            event({"kind": "error", "type": type(exc).__name__,
                                   "http_status": getattr(exc, "status_code", None)})
                            print(f"Stopped at task {row['task_id']}: {type(exc).__name__}. Attempt preserved; rerun to resume.", flush=True)
                            raise RuntimeError(f"Teacher run stopped: {type(exc).__name__}") from None
                    episode["attempt_file"] = str(attempt.relative_to(args.output_dir))
                    write_json(target, episode)
                    episodes.append(episode)
                    # Small summary every task; large exports every 25 tasks and at exit.
                    summary = summarize(args.output_dir, rows, episodes, export=len(episodes) % 25 == 0)
                    print(f"[{position}/{len(selected)}] task={row['task_id']} reward={episode['reward']:.3f} "
                          f"success={episode['success']} steps={len(episode['steps'])} "
                          f"total_completed={summary['completed_tasks']} sft={summary['clean_sft_trajectories']}", flush=True)
            finally:
                client.close()
                print(json.dumps(summarize(args.output_dir, rows), ensure_ascii=False, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
