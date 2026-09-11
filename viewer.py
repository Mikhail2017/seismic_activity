"""Gradio viewer for shot × receiver-line 2D seismic images."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np

from seismic_utils.dataset import (
    DEFAULT_DATA_DIR,
    LineGatherRef,
    ShotGather,
    build_line_gather_index,
    list_dataset_files,
    load_line_gather,
)
from seismic_utils.fb_smooth import DEFAULT_SMOOTH_THRESHOLD
from seismic_utils.hardpicks_bridge import HardpicksGatherStore, hardpicks_available
from seismic_utils.pickers import (
    ALL_PICKERS,
    PICKER_BEFORE_AFTER,
    PICKER_STA_LTA,
    normalize_picker as _registry_normalize_picker,
    picker_from_model,
    smooth_threshold_from_hparams,
)
from seismic_utils.plotting import plot_shot_gather
from seismic_utils.sites import SiteConfig, resolve_site_config

# Cache per resolved HDF5 path: (site, refs, optional lazy hardpicks store)
_INDEX_CACHE: dict[str, tuple[SiteConfig, list[LineGatherRef], HardpicksGatherStore | None]] = {}

# Loaded FBPUNet for prediction mode: (resolved ckpt path, model)
_MODEL_CACHE: dict[str, object] = {"path": None, "model": None}


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


def _default_ckpt() -> str:
    env = os.environ.get("FBP_CKPT_PATH", "").strip()
    if env:
        return env
    candidates = sorted(
        (Path(__file__).resolve().parent / "output").glob("train_*/best*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else ""


def _cache_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _prefer_hardpicks_loads() -> bool:
    """
    Whether reference gathers are loaded via hardpicks.

    Default is **native** (fast index + load). Set ``SEISMIC_BACKEND=hardpicks``
    to force official hardpicks loads for browsing (slower open). Prediction
    always uses hardpicks when available.
    """
    backend = os.environ.get("SEISMIC_BACKEND", "native").strip().lower()
    return backend in {"hardpicks", "official"}


def _get_index(path: Path):
    key = _cache_key(path)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    site = resolve_site_config(path)
    # Always build the dropdown with the native vectorized index (seconds, not minutes).
    index = build_line_gather_index(path, site=site)
    store: HardpicksGatherStore | None = None
    if _prefer_hardpicks_loads():
        if not hardpicks_available():
            raise gr.Error(
                "SEISMIC_BACKEND=hardpicks but hardpicks is not importable "
                "(need hardpicks + torch). Unset SEISMIC_BACKEND or install deps."
            )
        store = HardpicksGatherStore(path, site=site, defer_open=False)
    elif hardpicks_available():
        # Fast browsing: defer TraceParser until the first prediction.
        store = HardpicksGatherStore(path, site=site, defer_open=True, refs=index)
    _INDEX_CACHE[key] = (site, index, store)
    return site, index, store


def _gather_choice(ref: LineGatherRef) -> str:
    return ref.label


def _parse_gather_id(choice: str) -> int:
    return int(choice.split(":", 1)[0].strip())


PICKER_MODES = ALL_PICKERS


def _normalize_pick_display(pick_display: str) -> str:
    mode = (pick_display or "reference").strip().lower()
    if mode not in {"reference", "prediction", "both"}:
        raise gr.Error(f"Unknown pick display mode: {pick_display!r}")
    return mode


def _normalize_picker(picker: str) -> str:
    try:
        return _registry_normalize_picker(picker)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _smooth_threshold_from_ui(value) -> int:
    if value is None:
        return DEFAULT_SMOOTH_THRESHOLD
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return DEFAULT_SMOOTH_THRESHOLD


def _sta_lta_options_from_ui(th: float | None, lw_s: float | None, sw_s: float | None):
    from seismic_utils.sta_lta import StaLtaOptions

    return StaLtaOptions(
        th=float(th) if th is not None else 1.3,
        lw_s=float(lw_s) if lw_s is not None else 0.5,
        sw_s=float(sw_s) if sw_s is not None else 0.05,
    )


def load_model(ckpt_path: str):
    """Load / reload an FBPUNet checkpoint for prediction mode."""
    from seismic_utils.predict import load_fbp_model, resolve_checkpoint

    if not ckpt_path or not str(ckpt_path).strip():
        raise gr.Error("Provide a path to a .ckpt file (or a train_* dir with best*.ckpt).")

    path = Path(ckpt_path).expanduser()
    try:
        if path.is_dir():
            resolved = resolve_checkpoint(ckpt_dir=path)
        else:
            resolved = resolve_checkpoint(path)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    key = str(resolved)
    if _MODEL_CACHE["path"] == key and _MODEL_CACHE["model"] is not None:
        model = _MODEL_CACHE["model"]
        status = f"Model already loaded: {resolved.name}"
    else:
        try:
            model = load_fbp_model(resolved)
        except Exception as exc:  # noqa: BLE001 — surface to Gradio
            raise gr.Error(f"Failed to load checkpoint: {exc}") from exc
        _MODEL_CACHE["path"] = key
        _MODEL_CACHE["model"] = model
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        status = (
            f"Loaded {resolved} | {n_params / 1e6:.1f}M params | "
            f"device={next(model.parameters()).device}"
        )
    detected = picker_from_model(model)
    hp = dict(getattr(model, "hparams", {}) or {})
    thr = smooth_threshold_from_hparams(hp)
    status = f"{status} | picker={detected}"
    return (
        str(resolved),
        status,
        gr.update(value=detected),
        gr.update(value=thr),
    )


def _get_loaded_model():
    model = _MODEL_CACHE.get("model")
    if model is None:
        raise gr.Error("Load a checkpoint first (Prediction section → Load model).")
    return model


def _predict_for_gather(
    store: HardpicksGatherStore | None,
    ref: LineGatherRef,
    gather: ShotGather | None = None,
    *,
    picker: str = "fbpunet",
    sta_lta_opts=None,
    smooth_threshold: int | None = None,
) -> np.ndarray:
    picker = _normalize_picker(picker)
    if picker == PICKER_STA_LTA:
        from seismic_utils.sta_lta import pick_first_breaks_ms_from_shot_gather

        if gather is None:
            raise gr.Error("STA-LTA picking needs a loaded gather.")
        try:
            return pick_first_breaks_ms_from_shot_gather(gather, sta_lta_opts)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"STA-LTA pick failed: {exc}") from exc

    from seismic_utils.predict import (
        predict_first_breaks_ms,
        predict_first_breaks_ms_from_shot_gather,
    )

    model = _get_loaded_model()
    # Prefer the already-loaded native gather — avoids opening hardpicks TraceParser
    # (multi-minute hang on large HDF5 sites like Brunswick).
    if gather is not None:
        return predict_first_breaks_ms_from_shot_gather(
            model, gather, picker=picker, smooth_threshold=smooth_threshold
        )
    if store is None:
        raise gr.Error(
            "Prediction requires a loaded gather or hardpicks (+ torch). "
            "Install deps / run setup_lightning.sh."
        )
    try:
        raw = store.load_raw_by_shot_line(ref.shot_id, ref.line_id)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Failed to open gather for prediction: {exc}") from exc
    return predict_first_breaks_ms(
        model, raw, picker=picker, smooth_threshold=smooth_threshold
    )


def load_dataset(
    file_path: str,
    show_first_breaks: bool,
    highlight_regions: bool,
    pick_display: str = "reference",
    picker: str = "fbpunet",
    sta_lta_th: float | None = 1.3,
    sta_lta_lw: float | None = 0.5,
    sta_lta_sw: float | None = 0.05,
    smooth_threshold: float | None = DEFAULT_SMOOTH_THRESHOLD,
    force_reload: bool = False,
):
    path = Path(file_path).expanduser()
    if path.is_dir():
        files = list_dataset_files(path)
        if not files:
            raise gr.Error(f"No HDF5 files found in {path}")
        path = files[0]

    if not path.exists():
        raise gr.Error(f"File not found: {path}")

    key = _cache_key(path)
    was_cached = (not force_reload) and key in _INDEX_CACHE
    if force_reload:
        _INDEX_CACHE.pop(key, None)
    site, index, store = _get_index(path)
    if not index:
        raise gr.Error("No shot×receiver-line gathers found in the dataset.")

    choices = [_gather_choice(ref) for ref in index]
    figure = plot_selected(
        str(path),
        choices[0],
        show_first_breaks=show_first_breaks,
        highlight_regions=highlight_regions,
        pick_display=pick_display,
        picker=picker,
        sta_lta_th=sta_lta_th,
        sta_lta_lw=sta_lta_lw,
        sta_lta_sw=sta_lta_sw,
        smooth_threshold=smooth_threshold,
    )
    backend = "hardpicks" if _prefer_hardpicks_loads() and store is not None else "native"
    model_note = ""
    picker_mode = _normalize_picker(picker)
    if picker_mode == PICKER_STA_LTA:
        model_note = " | picker=STA-LTA-OS"
    else:
        bits = [f"picker={picker_mode}"]
        if picker_mode == PICKER_BEFORE_AFTER:
            bits.append(f"smooth={_smooth_threshold_from_ui(smooth_threshold)}")
        if _MODEL_CACHE["path"]:
            bits.append(f"model={Path(_MODEL_CACHE['path']).name}")
        model_note = " | " + " | ".join(bits)
    cache_note = "cache hit" if was_cached else "indexed"
    status = (
        f"Loaded {path.name} ({site.site_name}): {len(index)} line gathers | "
        f"backend={backend} ({cache_note}) | FB={site.first_break_field_name}, "
        f"REC_PEG digits={site.receiver_id_digit_count}{model_note}"
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
    pick_display: str = "reference",
    picker: str = "fbpunet",
    sta_lta_th: float | None = 1.3,
    sta_lta_lw: float | None = 0.5,
    sta_lta_sw: float | None = 0.05,
    smooth_threshold: float | None = DEFAULT_SMOOTH_THRESHOLD,
):
    if not file_path or not gather_choice:
        raise gr.Error("Select a dataset file and a line gather.")

    path = Path(file_path).expanduser()
    if not path.exists():
        raise gr.Error(f"File not found: {path}")

    mode = _normalize_pick_display(pick_display)
    picker_mode = _normalize_picker(picker)
    site, index, store = _get_index(path)
    gather_id = _parse_gather_id(gather_choice)
    if gather_id < 0 or gather_id >= len(index):
        raise gr.Error(f"Invalid gather id: {gather_id}")
    ref = index[gather_id]

    use_hardpicks_load = _prefer_hardpicks_loads() and store is not None and store.is_open
    if use_hardpicks_load:
        gather = store.load_by_shot_line(ref.shot_id, ref.line_id)
    else:
        gather = load_line_gather(path, gather_id=gather_id, site=site, index=index)

    pred_ms = None
    if mode in {"prediction", "both"}:
        pred_ms = _predict_for_gather(
            store,
            ref,
            gather=gather,
            picker=picker_mode,
            sta_lta_opts=_sta_lta_options_from_ui(sta_lta_th, sta_lta_lw, sta_lta_sw),
            smooth_threshold=_smooth_threshold_from_ui(smooth_threshold),
        )

    return plot_shot_gather(
        gather,
        show_first_breaks=show_first_breaks,
        highlight_regions=highlight_regions,
        predicted_first_breaks_ms=pred_ms,
        pick_display=mode,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Seismic Line Gather Viewer") as demo:
        gr.Markdown(
            "# Seismic Line Gather Viewer\n"
            "2D images are **shot × receiver-line** gathers. "
            "Browsing uses the **native** index/loader by default (fast). "
            "Set `SEISMIC_BACKEND=hardpicks` for official hardpicks loads (slower open).\n\n"
            "**Prediction mode:** choose **fbpunet** (FB-pixel UNet), **before_after** "
            "(horizon / before-vs-after UNet), or **STA-LTA-OS** (no checkpoint). "
            "Load a training `best*.ckpt` for the neural pickers — the radio switches to match "
            "the checkpoint. Then switch picks between reference labels, predictions (lime), or both. "
            "Neural-net predictions run on the already-loaded gather (no full hardpicks HDF5 open)."
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
            pick_display = gr.Radio(
                label="Pick display",
                choices=["reference", "prediction", "both"],
                value="reference",
                info="Prediction/both: load a checkpoint, or use STA-LTA (no ckpt).",
            )
            picker = gr.Radio(
                label="Picker",
                choices=list(PICKER_MODES),
                value="fbpunet",
                info="before_after uses the horizon smoother; sta-lta needs no checkpoint.",
            )

        with gr.Accordion("Prediction model", open=False):
            with gr.Row():
                ckpt_path = gr.Textbox(
                    label="Checkpoint path (.ckpt or train_* directory)",
                    value=_default_ckpt(),
                    scale=4,
                )
                load_model_btn = gr.Button("Load model", scale=1)
            model_status = gr.Textbox(label="Model status", interactive=False)
            smooth_threshold = gr.Number(
                label="Horizon smooth threshold (samples)",
                value=DEFAULT_SMOOTH_THRESHOLD,
                minimum=1,
                maximum=500,
                step=1,
                info="before_after picker: skip isolated after-class pixels until this many samples agree.",
            )

        with gr.Accordion("STA-LTA-OS", open=False):
            gr.Markdown(
                "Adaptive STA-LTA with outlier statistics (Jones & van der Baan, 2015). "
                "First-break mode fits a two-state HMM on the **full gather**, then picks the "
                "first onset with a short-window STA of outlier probability. "
                "**Th** and **short window** control the pick; the long window is the paper’s "
                "EM analysis length and is used to size the short window."
            )
            with gr.Row():
                sta_lta_th = gr.Number(label="Threshold Th", value=1.3, minimum=0.1, maximum=5.0, step=0.1)
                sta_lta_lw = gr.Number(label="Long window (s)", value=0.5, minimum=0.05, maximum=2.0, step=0.05)
                sta_lta_sw = gr.Number(label="Short window (s)", value=0.05, minimum=0.004, maximum=0.5, step=0.005)

        plot = gr.Plot(label="Line gather")

        plot_inputs = [
            file_path,
            gather_dropdown,
            show_fb,
            highlight_regions,
            pick_display,
            picker,
            sta_lta_th,
            sta_lta_lw,
            sta_lta_sw,
            smooth_threshold,
        ]
        load_inputs = [
            file_path,
            show_fb,
            highlight_regions,
            pick_display,
            picker,
            sta_lta_th,
            sta_lta_lw,
            sta_lta_sw,
            smooth_threshold,
        ]

        load_btn.click(
            fn=load_dataset,
            inputs=load_inputs,
            outputs=[file_path, gather_dropdown, plot, status],
        )
        load_model_btn.click(
            fn=load_model,
            inputs=[ckpt_path],
            outputs=[ckpt_path, model_status, picker, smooth_threshold],
        ).then(
            fn=plot_selected,
            inputs=plot_inputs,
            outputs=[plot],
        )
        gather_dropdown.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        show_fb.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        highlight_regions.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        pick_display.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        picker.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        sta_lta_th.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        sta_lta_lw.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        sta_lta_sw.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])
        smooth_threshold.change(fn=plot_selected, inputs=plot_inputs, outputs=[plot])

    return demo


if __name__ == "__main__":
    _bypass_proxy_for_localhost()
    build_app().launch(server_name="127.0.0.1", server_port=7860)
