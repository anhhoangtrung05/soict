"""Patch Sigma rules against verified evasion samples through Google AI Studio.

Drives a real Chromium browser with Playwright against the AI Studio
Playground (aistudio.google.com) - not the Gemini consumer chat UI, and not
a direct API call. AI Studio's free quota is a separate pool from the
consumer gemini.google.com UI, and the model is preselected via the page
URL's ?model= query param, so no model-picker automation is needed. A
persistent browser profile keeps the Google login across runs: you log in
by hand once, then this script sends one filled ``prompt_rule_patch_e2e.txt``
per rule, waits for the reply, and writes the patched Sigma YAML to disk.

Setup (one time):
    pip install playwright
    pip install pyyaml   # optional; a dependency-free structural check runs either way

    This drives your real installed Google Chrome (channel="chrome"), not
    Playwright's bundled Chromium, so `playwright install chromium` is not
    required. Google blocks sign-in from an automation-controlled window
    ("This browser or app may not be secure"), so do the first login by hand
    in a plain, non-automated Chrome window pointed at the same profile
    directory this script uses, then close that Chrome window completely
    before running the script:

        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ^
            --user-data-dir="<repo>\\.gemini_browser_profile"

    Log in and visit aistudio.google.com once there, then close every window
    of that Chrome instance. From then on this script reuses the
    already-authenticated profile and never performs the login itself.

Usage (run from the repository root):
    python -m src.generator.gemini_rule_patcher \
        --bypass-commands demo/backtest_bypass_commands.json \
        --rules-dir rules \
        --output-dir rules_patched
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback; see validate_sigma_yaml
    yaml = None

try:
    from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright
except ModuleNotFoundError:  # pragma: no cover - import guarded so --help works without Playwright
    Page = Any  # type: ignore[assignment]
    PlaywrightTimeoutError = TimeoutError  # type: ignore[assignment]
    sync_playwright = None


# /u/1/ ties this to whichever Google account is "account index 1" in your
# signed-in browser profile - override with --studio-url if that differs for you.
DEFAULT_STUDIO_URL = "https://aistudio.google.com/u/1/prompts/new_chat?pli=1&model=gemini-3.1-pro-preview"
DEFAULT_BYPASS_COMMANDS_PATH = Path("demo/backtest_bypass_commands.json")
DEFAULT_RULES_DIR = Path("rules")
DEFAULT_OUTPUT_DIR = Path("rules_patched")
DEFAULT_PROMPT_TEMPLATE = Path("prompt_rule_patch_e2e.txt")
DEFAULT_PROFILE_DIR = Path(".gemini_browser_profile")
DEFAULT_RAW_OUTPUT_DIR = Path("output/gemini_rule_patcher")
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 240.0
DEFAULT_STABLE_CHECKS = 3
DEFAULT_POLL_SECONDS = 1.5

SIGMA_PLACEHOLDER = "{{ORIGINAL_SIGMA_RULE}}"
EVASION_PLACEHOLDER = "{{EVASION_SAMPLES}}"

# Ordered fallback selectors. AI Studio's markup changes without notice, so
# every interaction tries each selector in turn instead of trusting one guess.
COMPOSER_SELECTORS = [
    "ms-autosize-textarea textarea",
    'textarea[placeholder*="Start typing a prompt" i]',
    'textarea[aria-label*="prompt" i]',
    "textarea",
]
RUN_BUTTON_SELECTORS = [
    'run-button button',
    'button:has-text("Run")',
    'button[aria-label*="Run" i]',
]
RESPONSE_CONTAINER_SELECTORS = [
    "ms-chat-turn .turn-content .markdown",
    "ms-model-turn .markdown",
    "[data-turn-role='model'] .markdown",
    "ms-chat-turn .markdown",
]
# Material Symbols icon buttons render their icon name as literal ligature
# text (e.g. "content_copy"), so matching on it works regardless of the UI
# language - unlike an aria-label, which is translated.
CLIPBOARD_COPY_BUTTON_SELECTORS = [
    'button:has-text("content_copy")',
    'button:has-text("copy_all")',
    "copy-button button",
    '[data-test-id="copy-button"]',
    'button[aria-label="Copy code"]',
    'button[aria-label="Copy"]',
]


class GeneratorError(RuntimeError):
    """Raised when the patch run cannot continue."""


@dataclass(frozen=True)
class PatchJob:
    """One Sigma rule and the evasion command lines it must be patched against."""

    rule_stem: str
    rule_path: Path
    rule_text: str
    bypass_commandlines: tuple[str, ...]
    seed_commandline: str
    technique_id: str


@dataclass(frozen=True)
class PatcherConfig:
    """Runtime settings for one patching run."""

    bypass_commands_path: Path = DEFAULT_BYPASS_COMMANDS_PATH
    rules_dir: Path = DEFAULT_RULES_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prompt_template: Path = DEFAULT_PROMPT_TEMPLATE
    profile_dir: Path = DEFAULT_PROFILE_DIR
    raw_output_dir: Path = DEFAULT_RAW_OUTPUT_DIR
    studio_url: str = DEFAULT_STUDIO_URL
    include_rules: tuple[str, ...] = ()
    limit_rules: int | None = None
    min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS
    headless: bool = False
    overwrite: bool = False
    resume: bool = True
    continue_on_error: bool = True
    dry_run: bool = False
    export_prompts: bool = False


def read_text(path: Path) -> str:
    """Read UTF-8 text and raise a clear error when the file is absent."""
    if not path.exists():
        raise GeneratorError(f"file not found: {path}")
    return path.read_text(encoding="utf-8-sig")


def write_text_overwrite(path: Path, text: str) -> None:
    """Write text atomically, overwriting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    temp_path.replace(path)


def normalize_rule_name(value: str) -> str:
    """Return a bare rule stem from either a path or a rule name."""
    text = str(value).strip().strip("\"'")
    if not text:
        raise GeneratorError("empty rule name")
    return Path(text).stem


def load_bypass_entries(path: Path) -> list[dict[str, Any]]:
    """Read the backtest bypass command list."""
    data = json.loads(read_text(path))
    if not isinstance(data, list):
        raise GeneratorError(f"{path} must contain a JSON array")
    return data


def build_jobs(config: PatcherConfig) -> list[PatchJob]:
    """Build one patch job per rule entry in the bypass command file."""
    entries = load_bypass_entries(config.bypass_commands_path)
    requested = {normalize_rule_name(rule) for rule in config.include_rules}

    jobs: list[PatchJob] = []
    missing_rule_files: list[str] = []
    for entry in entries:
        rule_stem = str(entry.get("target_rule") or "").strip()
        if not rule_stem:
            continue
        if requested and rule_stem not in requested:
            continue

        rule_path = config.rules_dir / f"{rule_stem}.yml"
        if not rule_path.exists():
            missing_rule_files.append(rule_stem)
            continue

        raw_bypass_commandlines = entry.get("bypass_commandlines") or []
        if isinstance(raw_bypass_commandlines, str):
            # Upstream serialization bug: entries with exactly one bypass
            # variant store it as a bare string instead of a one-item list.
            # Iterating a bare string in Python walks it character by
            # character, so this must be wrapped before the loop below.
            raw_bypass_commandlines = [raw_bypass_commandlines]
        bypass_commandlines = tuple(
            str(cmd).strip() for cmd in raw_bypass_commandlines if str(cmd).strip()
        )
        if not bypass_commandlines:
            continue

        jobs.append(
            PatchJob(
                rule_stem=rule_stem,
                rule_path=rule_path,
                rule_text=read_text(rule_path),
                bypass_commandlines=bypass_commandlines,
                seed_commandline=str(entry.get("seed_commandline") or ""),
                technique_id=str(entry.get("technique_id") or ""),
            )
        )

    if missing_rule_files:
        print(
            f"[!] Skipping {len(missing_rule_files)} rule(s) with no file under {config.rules_dir}: "
            f"{', '.join(missing_rule_files[:10])}{' ...' if len(missing_rule_files) > 10 else ''}",
            file=sys.stderr,
        )

    if config.limit_rules is not None:
        jobs = jobs[: config.limit_rules]
    return jobs


def format_evasion_samples(commandlines: Iterable[str]) -> str:
    """Format evasion command lines as a numbered list for the prompt."""
    commands = [cmd.strip() for cmd in commandlines if cmd.strip()]
    if not commands:
        raise GeneratorError("no evasion command lines to format")
    return "\n".join(f"{index}. {command}" for index, command in enumerate(commands, start=1))


def render_prompt(template: str, rule_text: str, bypass_commandlines: Iterable[str]) -> str:
    """Fill prompt_rule_patch_e2e.txt with one rule and its evasion samples."""
    if SIGMA_PLACEHOLDER not in template:
        raise GeneratorError(f"prompt template missing {SIGMA_PLACEHOLDER}")
    if EVASION_PLACEHOLDER not in template:
        raise GeneratorError(f"prompt template missing {EVASION_PLACEHOLDER}")
    return template.replace(SIGMA_PLACEHOLDER, rule_text).replace(
        EVASION_PLACEHOLDER, format_evasion_samples(bypass_commandlines)
    )


def strip_markdown_fence(text: str) -> str:
    """Remove a single leading/trailing ``` fence if the model added one anyway."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


YAML_FENCE_RE = re.compile(r"```(?:ya?ml)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_yaml_rule(response_text: str) -> str:
    """Extract the Sigma YAML document from a raw model response.

    The prompt asks for a single fenced ```yaml block (preferred: it also
    gives the UI a code block with a "copy" control for exact-text
    extraction). Older-style unfenced output - everything from the first
    'title:' line onward - is still accepted as a fallback.
    """
    fence_match = YAML_FENCE_RE.search(response_text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    text = strip_markdown_fence(response_text)
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("title:")), None)
    if start is None:
        raise GeneratorError("response does not contain a fenced YAML block or a 'title:' line")
    return "\n".join(lines[start:]).strip()


TOP_LEVEL_KEY_RE = {
    key: re.compile(rf"^{key}:") for key in ("title", "id", "detection", "logsource", "condition")
}


def validate_sigma_yaml(text: str) -> tuple[bool, str | None]:
    """Return whether text looks like a minimally valid Sigma rule.

    Always runs a dependency-free structural check (required top-level keys,
    a condition nested under detection). When PyYAML happens to be installed,
    it is also used for a stronger semantic parse — but it is optional, since
    PyYAML has no prebuilt wheel yet on some newer Python builds and this
    script must keep working without it.
    """
    lines = text.splitlines()
    if not lines:
        return False, "empty document"

    for key in ("title", "detection", "logsource"):
        if not any(TOP_LEVEL_KEY_RE[key].match(line) for line in lines):
            return False, f"missing top-level '{key}:'"
    if not any(re.match(r"^\s+condition:", line) for line in lines):
        return False, "missing 'condition:' nested under detection"

    if yaml is None:
        return True, None

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return False, f"invalid YAML: {exc}"
    if not isinstance(data, dict):
        return False, "parsed YAML is not a mapping"
    detection = data.get("detection")
    if not isinstance(detection, dict) or "condition" not in detection:
        return False, "detection block is empty or missing 'condition'"
    return True, None


def wait_for_request_slot(last_request_started_at: float | None, min_interval_seconds: float) -> float:
    """Throttle requests so AI Studio is not hit faster than min_interval_seconds apart."""
    now = time.monotonic()
    if last_request_started_at is None or min_interval_seconds <= 0:
        return now
    elapsed = now - last_request_started_at
    sleep_seconds = min_interval_seconds - elapsed
    if sleep_seconds > 0:
        print(f"    [rate-limit] waiting {sleep_seconds:.1f}s before next request", file=sys.stderr)
        time.sleep(sleep_seconds)
        now = time.monotonic()
    return now


def find_first_visible(page: Page, selectors: list[str], timeout_ms: int = 10000):
    """Return the first visible locator among several fallback selectors."""
    last_exc: Exception | None = None
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout_ms)
            return locator, selector
        except PlaywrightTimeoutError as exc:  # noqa: PERF203 - trying alternatives is the point
            last_exc = exc
            continue
    raise GeneratorError(f"none of these selectors matched: {selectors}") from last_exc


def find_last_visible(page: Page, selectors: list[str], timeout_ms: int = 10000):
    """Return the last (most recent) visible locator among fallback selectors.

    Used to target the newest response's copy button rather than one from an
    earlier turn in the conversation.
    """
    last_exc: Exception | None = None
    for selector in selectors:
        try:
            locator = page.locator(selector).last
            locator.wait_for(state="visible", timeout=timeout_ms)
            return locator, selector
        except PlaywrightTimeoutError as exc:  # noqa: PERF203 - trying alternatives is the point
            last_exc = exc
            continue
    raise GeneratorError(f"none of these selectors matched: {selectors}") from last_exc


def wait_for_manual_setup(page: Page) -> None:
    """Pause once so the user can log in by hand."""
    print(
        "\n"
        "[setup] A Chrome window just opened at aistudio.google.com, with\n"
        "        Gemini 3.1 Pro Preview preselected via the page URL.\n"
        "        If you already logged in by hand in a separate, non-automated Chrome\n"
        "        window pointed at this same profile directory (see the module docstring),\n"
        "        this window should already be signed in - just confirm the page loaded.\n"
        "        If this window is NOT signed in, do not log in here: Google blocks sign-in\n"
        "        from an automated window. Close this run, log in by hand in a plain Chrome\n"
        "        window using the same profile directory, then rerun this script.\n",
        file=sys.stderr,
    )
    input("Press Enter here once you have confirmed the page is loaded and signed in... ")


def submit_prompt(page: Page, prompt_text: str) -> None:
    """Type the prompt into AI Studio's composer and run it.

    Unlike Gemini's chat box, this composer is a plain textarea: Enter
    inserts a newline and Ctrl+Enter (or the Run button) submits.
    """
    composer, _ = find_first_visible(page, COMPOSER_SELECTORS, timeout_ms=20000)
    composer.click()
    lines = prompt_text.split("\n")
    for index, line in enumerate(lines):
        if line:
            page.keyboard.insert_text(line)
        if index != len(lines) - 1:
            page.keyboard.press("Enter")
    try:
        run_button, _ = find_first_visible(page, RUN_BUTTON_SELECTORS, timeout_ms=5000)
        run_button.click()
    except GeneratorError:
        page.keyboard.press("Control+Enter")


def wait_for_response(
    page: Page,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    stable_checks: int = DEFAULT_STABLE_CHECKS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> str:
    """Poll the last response container until its text stops changing (streaming done)."""
    container_selector: str | None = None
    deadline = time.monotonic() + max_wait_seconds

    while container_selector is None and time.monotonic() < deadline:
        for selector in RESPONSE_CONTAINER_SELECTORS:
            if page.locator(selector).count() > 0:
                container_selector = selector
                break
        if container_selector is None:
            page.wait_for_timeout(1000)

    if container_selector is None:
        raise GeneratorError("could not locate an AI Studio response container; the page layout may have changed")

    last_text: str | None = None
    stable_count = 0
    while time.monotonic() < deadline:
        try:
            text = page.locator(container_selector).last.inner_text()
        except Exception:  # noqa: BLE001 - DOM can detach mid-render; just retry
            text = ""
        if text and text == last_text:
            stable_count += 1
            if stable_count >= stable_checks:
                return text
        else:
            stable_count = 0
        last_text = text
        page.wait_for_timeout(int(poll_seconds * 1000))

    if last_text:
        print("    [!] response wait timed out; using the last captured text", file=sys.stderr)
        return last_text
    raise GeneratorError("AI Studio did not return any response text before the timeout")


def extract_exact_response_text(page: Page, fallback_text: str) -> str:
    """Return the response's exact source text, preferring the clipboard over inner_text().

    Rendered code blocks commonly apply indentation visually (CSS) rather
    than with literal leading-space characters in the DOM text, so
    .inner_text() can silently drop all indentation - fatal for YAML, where
    nesting is whitespace-only. Clicking the page's own "copy code" control
    re-serializes the original raw text into the OS clipboard, which this
    reads back directly.
    """
    try:
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    except Exception:  # noqa: BLE001 - permission grant can fail per-browser; clipboard read below will too
        pass

    try:
        copy_button, _ = find_last_visible(page, CLIPBOARD_COPY_BUTTON_SELECTORS, timeout_ms=8000)
        copy_button.click()
        page.wait_for_timeout(500)
        clipboard_text = page.evaluate("navigator.clipboard.readText()")
        if clipboard_text and clipboard_text.strip():
            return clipboard_text
        print("    [!] clipboard was empty after clicking copy; falling back to rendered text", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - degrade to the (possibly unindented) rendered text instead of failing the rule
        print(
            f"    [!] could not read the response via clipboard copy ({exc}); "
            "falling back to rendered text, which may be missing indentation",
            file=sys.stderr,
        )
    return fallback_text


def export_prompts(config: PatcherConfig, jobs: list[PatchJob], template: str) -> int:
    """Write one rules_patched/<rule>/prompt.txt per job that still needs work.

    For a manual send-by-hand workflow: no browser involved. Rules that
    already have a successful flat rules_patched/<rule>.yml are skipped
    (unless --overwrite), so this only touches the ones still needing a
    prompt. The matching rules_patched/<rule>/<rule>.yml is added later,
    once a response is pasted back for that rule.
    """
    written = 0
    skipped = 0
    for job in jobs:
        flat_output_path = config.output_dir / f"{job.rule_stem}.yml"
        if config.resume and flat_output_path.exists() and not config.overwrite:
            skipped += 1
            continue
        prompt = render_prompt(template, job.rule_text, job.bypass_commandlines)
        prompt_path = config.output_dir / job.rule_stem / "prompt.txt"
        write_text_overwrite(prompt_path, prompt)
        print(f"    [+] wrote {prompt_path} ({len(job.bypass_commandlines)} evasion sample(s))")
        written += 1
    print(f"\n[+] Done: {written} prompt(s) written, {skipped} rule(s) already patched and skipped")
    return 0


def run_patcher(config: PatcherConfig) -> int:
    """Run the full patch loop against AI Studio."""
    template = read_text(config.prompt_template)
    jobs = build_jobs(config)
    if not jobs:
        raise GeneratorError("no patch jobs found; check --bypass-commands and --rules-dir")

    if config.dry_run:
        print(f"[+] Dry run: {len(jobs)} rule(s) would be patched")
        for job in jobs:
            print(f"    - {job.rule_stem} ({len(job.bypass_commandlines)} evasion sample(s))")
            render_prompt(template, job.rule_text, job.bypass_commandlines)  # validate template placeholders
        return 0

    if config.export_prompts:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        return export_prompts(config, jobs, template)

    if sync_playwright is None:
        raise GeneratorError("playwright is not installed; run: pip install playwright")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.raw_output_dir.mkdir(parents=True, exist_ok=True)
    config.profile_dir.mkdir(parents=True, exist_ok=True)

    run_log: list[dict[str, Any]] = []
    last_request_started_at: float | None = None

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            channel="chrome",  # drive your real installed Chrome, not Playwright's bundled Chromium
            headless=config.headless,
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(config.studio_url, wait_until="domcontentloaded")
        wait_for_manual_setup(page)

        print(f"[+] Patching {len(jobs)} rule(s) via AI Studio ({config.studio_url})")
        for index, job in enumerate(jobs, start=1):
            out_path = config.output_dir / f"{job.rule_stem}.yml"
            if config.resume and out_path.exists() and not config.overwrite:
                print(f"[{index}/{len(jobs)}] {job.rule_stem} [skip existing]")
                continue

            print(f"[{index}/{len(jobs)}] {job.rule_stem}: {len(job.bypass_commandlines)} evasion sample(s)")
            prompt = render_prompt(template, job.rule_text, job.bypass_commandlines)
            log_entry: dict[str, Any] = {"rule": job.rule_stem, "technique_id": job.technique_id}

            last_request_started_at = wait_for_request_slot(
                last_request_started_at, config.min_request_interval_seconds
            )
            try:
                page.goto(config.studio_url, wait_until="domcontentloaded", timeout=30000)
                submit_prompt(page, prompt)
                rendered_text = wait_for_response(page, max_wait_seconds=config.max_wait_seconds)
                response_text = extract_exact_response_text(page, rendered_text)
            except Exception as exc:  # noqa: BLE001 - keep going across rules unless told not to
                print(f"    [!] {exc}", file=sys.stderr)
                log_entry["error"] = str(exc)
                run_log.append(log_entry)
                if config.continue_on_error:
                    continue
                context.close()
                raise

            write_text_overwrite(config.raw_output_dir / f"{job.rule_stem}.response.txt", response_text)

            try:
                patched_yaml = extract_yaml_rule(response_text)
            except GeneratorError as exc:
                print(f"    [!] {exc}", file=sys.stderr)
                log_entry["error"] = str(exc)
                run_log.append(log_entry)
                continue

            ok, error = validate_sigma_yaml(patched_yaml)
            if not ok:
                print(f"    [!] invalid patched rule: {error}", file=sys.stderr)
                write_text_overwrite(config.output_dir / "_failed" / f"{job.rule_stem}.yml", patched_yaml)
                log_entry["error"] = error
                run_log.append(log_entry)
                continue

            write_text_overwrite(out_path, patched_yaml)
            print(f"    [+] wrote {out_path}")
            log_entry["output_path"] = str(out_path)
            run_log.append(log_entry)

            stale_failed_path = config.output_dir / "_failed" / f"{job.rule_stem}.yml"
            if stale_failed_path.exists():
                stale_failed_path.unlink()

        context.close()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_text_overwrite(
        config.raw_output_dir / f"run_{timestamp}.json",
        json.dumps({"created_at": timestamp, "jobs": run_log}, indent=2, ensure_ascii=False),
    )

    failed = sum(1 for entry in run_log if "error" in entry)
    succeeded = len(run_log) - failed
    print(f"\n[+] Done: {succeeded} patched, {failed} failed, {len(jobs) - len(run_log)} skipped")
    return 1 if failed and not config.continue_on_error else 0


def positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the AI Studio rule patcher."""
    parser = argparse.ArgumentParser(description="Patch Sigma rules against evasion samples using Gemini 3.1 Pro on Google AI Studio.")
    parser.add_argument("--bypass-commands", default=str(DEFAULT_BYPASS_COMMANDS_PATH))
    parser.add_argument("--rules-dir", default=str(DEFAULT_RULES_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT_TEMPLATE))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR), help="Persistent Chrome profile folder; keeps your Google login between runs.")
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR))
    parser.add_argument("--studio-url", default=DEFAULT_STUDIO_URL, help="AI Studio prompt URL, with ?model= selecting the model.")
    parser.add_argument("--rule", action="append", default=[], help="Rule stem to patch. Can be repeated.")
    parser.add_argument("--limit-rules", type=positive_int)
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        help="Minimum time between prompts sent to AI Studio, to avoid rate limiting/throttling.",
    )
    parser.add_argument("--max-wait-seconds", type=float, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument("--headless", action="store_true", help="Run Chromium headless. Google may challenge headless sessions; prefer headed.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Re-patch rules even if rules_patched/<rule>.yml already exists.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the whole run on the first failed rule instead of continuing.")
    parser.add_argument("--dry-run", action="store_true", help="List the jobs that would run without opening a browser.")
    parser.add_argument(
        "--export-prompts",
        action="store_true",
        help="Write rules_patched/<rule>/prompt.txt for each rule still needing work, instead of opening a browser.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PatcherConfig:
    """Create PatcherConfig from parsed CLI args."""
    if args.min_request_interval_seconds < 0:
        raise GeneratorError("--min-request-interval-seconds must be >= 0")
    if args.max_wait_seconds <= 0:
        raise GeneratorError("--max-wait-seconds must be > 0")

    return PatcherConfig(
        bypass_commands_path=Path(args.bypass_commands),
        rules_dir=Path(args.rules_dir),
        output_dir=Path(args.output_dir),
        prompt_template=Path(args.prompt_template),
        profile_dir=Path(args.profile_dir),
        raw_output_dir=Path(args.raw_output_dir),
        studio_url=args.studio_url,
        include_rules=tuple(args.rule),
        limit_rules=args.limit_rules,
        min_request_interval_seconds=args.min_request_interval_seconds,
        max_wait_seconds=args.max_wait_seconds,
        headless=args.headless,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        continue_on_error=not args.stop_on_error,
        dry_run=args.dry_run,
        export_prompts=args.export_prompts,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_patcher(config_from_args(args))
    except Exception as exc:  # noqa: BLE001 - surface a clean one-line error to the CLI user
        print(f"[!] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
