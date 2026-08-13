"""String formatting, pretty-print, and key-matching utilities."""

import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Union


def pretty_print_nested(
    cfg: dict,
    *,
    title: str = None,
    indent: int = 0,
    pad_before: bool = False,
    pad_after: bool = False,
):
    """Pretty-print a nested config dict in a compact runtime-summary layout."""
    inner_width = 80
    KEY_W = 21  # key column width for key:value lines
    lines: List[str] = []
    path_value_patterns = (
        r"[^/]+(?:/|$)",
        r"[^_]+(?:_|$)",
        r"[^-]+(?:-|$)",
    )
    enabled_states = {"enabled", "true", "active"}
    disabled_states = {"disabled", "false", "inactive"}

    class ANSI:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        YELLOW = "\033[33m"
        GREEN = "\033[32m"
        RED = "\033[31m"

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def visible_len(text: str) -> int:
        return len(ansi_re.sub("", text))

    def emit(text: str = "") -> None:
        lines.append(text)

    def is_checkpoint_path(key: str, path: tuple[str, ...]) -> bool:
        return (key.lower() in ("path", "ckpt_path", "checkpoint_path", "resume")) and (
            "Checkpoint" in path
        )

    def is_inline_view_block(d: dict) -> bool:
        return {"Mode", "Input Res", "Output Res"}.issubset(set(d.keys()))

    def titleize_label(label: Any) -> str:
        label_str = str(label)
        if "_" not in label_str and any(ch.isupper() for ch in label_str[1:]):
            return label_str
        return label_str.replace("_", " ").title()

    def normalize_status(status: Any) -> str:
        return str(status).strip().lower()

    def colorize(text: str, color: str, *, bold: bool = False) -> str:
        prefix = ANSI.BOLD if bold else ""
        return f"{prefix}{color}{text}{ANSI.RESET}"

    def status_parts(status: Any):
        status_str = normalize_status(status)
        if status_str in enabled_states:
            return ("✓", ANSI.GREEN)
        if status_str in disabled_states:
            return ("×", ANSI.RED)
        return ("•", None)

    def status_icon(status: Any) -> str:
        mark, color = status_parts(status)
        if color is None:
            return "[•]"
        return colorize(f"[{mark}]", color)

    def status_mark(status: Any) -> str:
        mark, color = status_parts(status)
        if color is None:
            return mark
        return colorize(mark, color)

    def format_action_rep(value: Any) -> str:
        current = normalize_status(value)

        def render_mode(mode_key: str) -> str:
            mode_label = str(mode_key).strip().replace("_", " ").title()
            is_selected = current == mode_key
            marker = status_icon("enabled" if is_selected else "disabled")
            if is_selected:
                return f"{marker} {colorize(mode_label, ANSI.GREEN, bold=True)}"
            return f"{marker} {mode_label}"

        return " | ".join(
            [
                render_mode("absolute"),
                render_mode("delta"),
                render_mode("relative"),
            ]
        )

    def format_inline_view(d: dict) -> str:
        mode = str(d.get("Mode"))
        input_res = str(d.get("Input Res"))
        output_res = str(d.get("Output Res"))
        return f"{mode}   ({input_res} -> {output_res})"

    def humanize_scalar_value(v: Any, *, key: str) -> Any:
        if isinstance(v, str):
            value = v.strip()
            lower = normalize_status(value)
            key_lower = str(key).strip().lower()

            if key_lower in {
                "action mode",
                "action rep",
                "action representation",
                "action chunk representation",
            }:
                return format_action_rep(value)

            if lower in enabled_states | disabled_states:
                return f"{status_icon(lower)} {value.title()}"

        return v

    def format_value(v, *, key: str, path: tuple[str, ...]):
        # ---- key-aware None handling
        if v is None:
            if is_checkpoint_path(key, path):
                return "Randomly initialized"
            return "None"

        # ---- empty containers
        if isinstance(v, (list, tuple)) and len(v) == 0:
            return "None"

        # ---- floats (generic)
        if isinstance(v, float):
            return f"{v:.3f}"

        return humanize_scalar_value(v, key=key)

    def wrap_value_text(text: str, max_width: int, *, use_path_wrap: bool = False) -> List[str]:
        if visible_len(text) <= max_width:
            return [text]

        # Prefer splitting path-like values at separators before hard wrapping.
        if use_path_wrap and any(sep in text for sep in ("/", "_", "-")):
            for pattern in path_value_patterns:
                tokens = re.findall(pattern, text)
                if len(tokens) <= 1:
                    continue
                if max(len(token.rstrip()) for token in tokens) > max_width:
                    continue

                lines: List[str] = []
                current = ""
                for token in tokens:
                    token = token.rstrip()
                    if not current:
                        current = token
                        continue
                    if len(current) + len(token) <= max_width:
                        current += token
                    else:
                        lines.append(current)
                        current = token
                if current:
                    lines.append(current)
                if len(lines) > 1:
                    return lines

        return textwrap.wrap(
            text,
            width=max_width,
            break_long_words=True,
            break_on_hyphens=True,
        )

    def get_wrapped_value_lines(
        value: Any,
        *,
        key: str,
        path: tuple[str, ...],
        max_width: int,
    ) -> List[str]:
        formatted = format_value(value, key=key, path=path)
        text = str(formatted)
        return wrap_value_text(
            text,
            max_width,
            use_path_wrap=is_checkpoint_path(key, path) or ("/" in text),
        )

    def print_kv(k: str, v, ind: int, path: tuple[str, ...]):
        value_width = max(8, inner_width - ind - KEY_W - 2)  # account for ": "
        wrapped = get_wrapped_value_lines(v, key=k, path=path, max_width=value_width)

        key_label = titleize_label(k)
        emit(" " * ind + f"{key_label:<{KEY_W}s}: {wrapped[0]}")
        for extra in wrapped[1:]:
            emit(" " * ind + " " * (KEY_W + 2) + extra)

    def print_heading(label: str, ind: int, *, status: Any = None):
        heading_label = titleize_label(label)
        heading = (
            f"[ {heading_label} ]"
            if status is None
            else f"[{status_mark(status)} {heading_label}]"
        )
        emit(" " * ind + heading)

    def print_bullet_heading(label: str, ind: int, *, status: Any = None):
        heading_label = titleize_label(label)
        heading = heading_label if status is None else f"{status_icon(status)} {heading_label}"
        emit(" " * ind + heading)

    def emit_labeled_block(label: str, ind: int) -> None:
        emit((" " * ind) + titleize_label(label))

    def render(d: dict, ind: int, path: tuple[str, ...]):
        first = True
        for k, v in d.items():
            if isinstance(v, dict):
                is_top_level = len(path) == 0
                if is_top_level and not first:
                    emit()  # blank line between top-level blocks only
                if is_top_level:
                    status = v.get("Status") if "Status" in v and len(v) > 1 else None
                    print_heading(k, ind, status=status)
                    rest = (
                        {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "Status"}
                        if status is not None
                        else v
                    )
                    render(rest, ind, path + (k,))
                elif is_inline_view_block(v):
                    print_kv(k, format_inline_view(v), ind + 2, path)
                elif "Status" in v and len(v) > 1:
                    if not first:
                        emit()
                    print_bullet_heading(k, ind + 2, status=v.get("Status"))
                    rest = {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "Status"}
                    render(rest, ind + 4, path + (k,))
                else:
                    emit_labeled_block(k, ind + 2)
                    render(v, ind + 4, path + (k,))
                first = False
                continue

            # ---- special case: list[str] rendered as a block
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                emit_labeled_block(k, ind + 2)
                for line in v:
                    emit(" " * (ind + 4) + line)
                first = False
                continue

            if len(path) == 0:
                if not first:
                    emit()
                print_heading(k, ind)
                wrapped = get_wrapped_value_lines(
                    v,
                    key=k,
                    path=path,
                    max_width=max(8, inner_width - (ind + 2)),
                )
                for line in wrapped:
                    emit(" " * (ind + 2) + line)
                first = False
                continue

            # ---- normal key:value
            print_kv(k, v, ind + 2, path)
            first = False

    if pad_before:
        print()

    emit()
    render(cfg, indent, path=())
    emit()

    frame_width = inner_width + 4
    if title is not None:
        title_text = f" {title} "
        pad = max(0, frame_width - 2 - len(title_text))
        top = "┌" + ("─" * (pad // 2)) + title_text + ("─" * (pad - pad // 2)) + "┐"
    else:
        top = "┌" + ("─" * (frame_width - 2)) + "┐"
    bottom = "└" + ("─" * (frame_width - 2)) + "┘"

    print(top)
    for line in lines:
        padding = max(0, inner_width - visible_len(line))
        print("│ " + line + (" " * padding) + " │")
    print(bottom)

    if pad_after:
        print()


def fold_path_from_marker(path: str, marker: str = "experiments") -> str:
    """Fold an absolute checkpoint path into a compact run identifier."""
    norm_path = os.path.normpath(str(path))
    parts = norm_path.split(os.sep)

    # Prefer showing path content after the marker (e.g. "experiments/...").
    if marker in parts:
        idx = parts.index(marker)
        folded = os.sep.join(parts[idx + 1 :])
    else:
        folded = norm_path

    # Drop the default checkpoint suffix from display.
    for ckpt_name in ("latest.ckpt",):
        default_suffix = os.path.join("checkpoints", ckpt_name)
        if folded.endswith(default_suffix):
            folded = folded[: -len(default_suffix)].rstrip(os.sep)
            break

    return folded


def divider(
    title: str = "", char: str = "-", width: int = 79, show: bool = True
) -> str:
    """
    Creates a horizontal divider line with an optional centered title.

    Args:
        title (str): Optional title to center in the divider.
        char (str): Character to use for the divider line.
        width (int): Total width of the divider.
        show (bool): If True, prints the divider directly. If False, returns the string.

    Returns:
        str: The formatted divider line (if show=False).
    """
    clean_title = title.strip()
    line = f" {clean_title} " if clean_title else ""
    side_len = (width - len(line)) // 2
    full_line = f"{char * side_len}{line}{char * (width - side_len - len(line))}"

    if show:
        print(full_line)
    else:
        return full_line


def plain_divider(char: str = "-", width: int = 79, show: bool = True) -> str:
    """
    Prints a plain horizontal divider.

    Args:
        char (str): Character to repeat.
        width (int): Width of the divider.
        show (bool): Whether to print or return the string.

    Returns:
        str: The divider string (if show=False).
    """
    line = char * width
    if show:
        print(line)
    else:
        return line


def print_dict(d: dict, title: str = "", indent: int = 0) -> None:
    """
    Prints a dictionary in a formatted way.

    Args:
        d (dict): The dictionary to print.
        title (str): Optional title for the dictionary.
        indent (int): Number of spaces to indent the output.
    """
    if title:
        print(f"{' ' * indent}{title}:")
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' ' * (indent + 2)}{key}:")
            print_dict(value, indent=indent + 4)
        else:
            print(f"{' ' * (indent + 2)}{key}: {value}")


def is_matched_key(pattern, key, reject_pattern=None):
    is_matched = re.search(rf"(^|_)({re.escape(pattern)})(_|$)", key)
    is_rejected = False
    if reject_pattern:
        is_rejected = re.search(rf"(^|_)({re.escape(reject_pattern)})(_|$)", key)
    return is_matched and not is_rejected


def find_all_keys(
    patterns: Union[str, List[str]],
    keys: Union[List[str], Dict[str, Any]],
    reject: Optional[List[str]] = None,
    strict: bool = True,
    only_one_match: bool = False,
) -> List[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    if reject is None:
        reject = []

    matched = []
    for key in keys:
        for pat in patterns:
            if strict:
                # match with underscore boundaries or start/end of string
                if re.search(rf"(^|_)({re.escape(pat)})(_|$)", key):
                    matched.append(key)
                    break
            else:
                if pat in key:
                    matched.append(key)
                    break

    for k in matched:
        for r in reject:
            if re.search(rf"(^|_)({re.escape(r)})(_|$)", k):
                matched.remove(k)
                break

    if only_one_match:
        if len(matched) == 0:
            return None
        if len(matched) > 1:
            raise ValueError(
                f"Multiple matches found: {matched}. Expected only one match."
            )
        return matched[0]
    return matched
