"""Gradio viewer for shot × receiver-line 2D seismic images."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from seismic_utils.dataset import (
    DEFAULT_DATA_DIR,
    LineGatherRef,
    build_line_gather_index,
    list_dataset_files,
    load_line_gather,
)
from seismic_utils.hardpicks_bridge import HardpicksGatherStore, hardpicks_available
from seismic_utils.plotting import plot_shot_gather
from seismic_utils.sites import SiteConfig, resolve_site_config

# Cache per resolved HDF5 path: (site, refs, optional hardpicks store)
_INDEX_CACHE: dict[str, tuple[SiteConfig, list[LineGatherRef], HardpicksGatherStore | None]] = {}


def _bypass_proxy_for_localhost() -> None:
    hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        existing = [p.strip() for p in os.environ.get(key, "").split(",") if p.strip()]
        merged = existing + [h for h in hosts if h not in existing]
        os.environ[key] = ",".join(merged)


def _default_file() -> str:
    files = list_dataset_files(DEFAULT_DATA_DIR)
    if not files:
        return str(DEFAULT_DATA_DIR)
    for path in files:
        if path.suffix.lower() != ".xz":
            return str(path)
    return str(files[0])


def _cache_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _prefer_hardpicks() -> bool:
    # Default backend is hardpicks. Set SEISMIC_BACKEND=native to use seismic_utils.
    backend = os.environ.get("SEISMIC_BACKEND", "hardpicks").strip().lower()
    if backend in {"native", "seismic_utils"}:
        return False
    return True


def _get_index(path: Path):
    key = _cache_key(path)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    site = resolve_site_config(path)
    store: HardpicksGatherStore | None = None
    if _prefer_hardpicks():
        if not hardpicks_available():
            raise gr.Error(
                "Default backend is hardpicks, but it is not importable "
                "(need hardpicks + torch). Install deps or set SEISMIC_BACKEND=native."
            )
        store = HardpicksGatherStore(path, site=site)
        index = store.refs
    else:
        index = build_line_gather_index(path, site=site)
    _INDEX_CACHE[key] = (site, index, store)
    return site, index, store


def _gather_choice(ref: LineGatherRef) -> str:
    return ref.label


def _parse_gather_id(choice: str) -> int:
    return int(choice.split(":", 1)[0].strip())


def load_dataset(file_path: str, show_first_breaks: bool, highlight_regions: bool):
    path = Path(file_path).expanduser()
    if path.is_dir():
        files = list_dataset_files(path)
        if not files:
            raise gr.Error(f"No HDF5 files found in {path}")
        path = files[0]

    if not path.exists():
        raise gr.Error(f"File not found: {path}")

    _INDEX_CACHE.pop(_cache_key(path), None)
    site, index, store = _get_index(path)
    if not index:
        raise gr.Error("No shot×receiver-line gathers found in the dataset.")

    choices = [_gather_choice(ref) for ref in index]
    figure = plot_selected(
        str(path),
        choices[0],
        show_first_breaks=show_first_breaks,
        highlight_regions=highlight_regions,
    )
    backend = "hardpicks" if store is not None else "native"
    status = (
        f"Loaded {path.name} ({site.site_name}): {len(index)} line gathers | "
        f"backend={backend} | FB={site.first_break_field_name}, "
        f"REC_PEG digits={site.receiver_id_digit_count}"
    )
    return (
        str(path),
        gr.update(choices=choices, value=choices[0]),
        figure,
        status,
    )


def plot_selected(
    file_path: str,
    gather_choice: str,
    show_first_breaks: bool = True,
    highlight_regions: bool = False,
):
    if not file_path or not gather_choice:
        raise gr.Error("Select a dataset file and a line gather.")

    path = Path(file_path).expanduser()
    if not path.exists():
        raise gr.Error(f"File not found: {path}")

    site, index, store = _get_index(path)
    gather_id = _parse_gather_id(gather_choice)
    if store is not None:
        gather = store.load(gather_id)
    else:
        gather = load_line_gather(path, gather_id=gather_id, site=site, index=index)
    return plot_shot_gather(
        gather,
        show_first_breaks=show_first_breaks,
        highlight_regions=highlight_regions,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Seismic Line Gather Viewer") as demo:
        gr.Markdown(
            "# Seismic Line Gather Viewer\n"
            "2D images are **shot × receiver-line** gathers (hardpicks parser by default). "
            "Set `SEISMIC_BACKEND=native` to use the local splitter."
        )

        with gr.Row():
            file_path = gr.Textbox(
                label="Dataset path",
                value=_default_file(),
                scale=4,
            )
            load_btn = gr.Button("Load dataset", variant="primary", scale=1)

        status = gr.Textbox(label="Status", interactive=False)
        gather_dropdown = gr.Dropdown(label="Line gather", choices=[], interactive=True)
        with gr.Row():
            show_fb = gr.Checkbox(label="Show first-break picks", value=True)
            highlight_regions = gr.Checkbox(
                label="Highlight before (blue) / after (red) first break",
                value=False,
            )
        plot = gr.Plot(label="Line gather")

        plot_inputs = [file_path, gather_dropdown, show_fb, highlight_regions]

        load_btn.click(
            fn=load_dataset,
            inputs=[file_path, show_fb, highlight_regions],
            outputs=[file_path, gather_dropdown, plot, status],
        )
        gather_dropdown.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        show_fb.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        highlight_regions.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])

    return demo


if __name__ == "__main__":
    _bypass_proxy_for_localhost()
    build_app().launch(server_name="127.0.0.1", server_port=7860)
