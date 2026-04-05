"""
Gradio: ảnh gốc sạch; chỉ khi hover vào đúng vùng bbox mới hiện khung + tooltip (HTML/CSS).

Notebook có thể dùng build_hover_figure (Plotly): tooltip khi hover vùng, box luôn mờ
(hoặc tắt hẳn viền — xem tham số).
"""

from __future__ import annotations

import argparse
import base64
import html as html_module
import uuid
from pathlib import Path

import cv2
import gradio as gr

from .config import OUTPUT_DIR
from .yolo_preview import parse_data_yaml_names, yolo_lines_to_boxes

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore


def _load_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_hover_review_html(
    image_path: Path,
    label_path: Path | None,
    class_names: list[str] | None,
) -> str:
    """
    Ảnh không vẽ box sẵn. Các lớp phủ trong suốt; :hover mới hiện viền cam + nền mờ.
    Tooltip: thuộc tính title của trình duyệt.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    uid = uuid.uuid4().hex[:10]
    parts: list[str] = [
        f"<style>"
        f".ab-wrap-{uid} {{ position: relative; display: inline-block; max-width: 100%; }}"
        f".ab-wrap-{uid} img {{ display: block; max-width: 100%; height: auto; vertical-align: top; }}"
        f".ab-wrap-{uid} .ab-hit-{uid} {{"
        f"  position: absolute; box-sizing: border-box;"
        f"  border: 2px solid transparent; background: transparent;"
        f"  pointer-events: auto;"
        f"}}"
        f".ab-wrap-{uid} .ab-hit-{uid}:hover {{"
        f"  border-color: rgba(255, 140, 0, 0.95);"
        f"  background: rgba(255, 165, 0, 0.2);"
        f"  z-index: 50;"
        f"}}"
        f"</style>"
        f'<div class="ab-wrap-{uid}">'
        f'<img src="data:image/jpeg;base64,{b64}" alt="frame" />'
    ]

    if label_path and label_path.is_file():
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        boxes = yolo_lines_to_boxes(lines, w, h) if lines else []
        for cls_id, x1, y1, x2, y2 in boxes:
            if class_names and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = f"class_{cls_id}"
            title = (
                f"{name} | id={cls_id} | "
                f"x1,y1={x1:.0f},{y1:.0f} | x2,y2={x2:.0f},{y2:.0f} | "
                f"{x2 - x1:.0f}x{y2 - y1:.0f} px"
            )
            left_pct = 100.0 * max(0, x1) / w
            top_pct = 100.0 * max(0, y1) / h
            width_pct = 100.0 * max(0, x2 - x1) / w
            height_pct = 100.0 * max(0, y2 - y1) / h
            t_attr = html_module.escape(title, quote=True)
            parts.append(
                f'<div class="ab-hit-{uid}" '
                f'style="left:{left_pct:.4f}%;top:{top_pct:.4f}%;width:{width_pct:.4f}%;height:{height_pct:.4f}%;" '
                f'title="{t_attr}"></div>'
            )

    parts.append("</div>")
    parts.append(
        f'<p style="opacity:0.85;font-size:0.9em;margin-top:8px;">'
        f"{html_module.escape(image_path.name)} — "
        f"Di chuột vào vùng có nhãn để hiện khung (không hover thì ảnh không có box).</p>"
    )
    return "".join(parts)


def build_hover_figure(
    image_path: Path,
    label_path: Path | None,
    class_names: list[str] | None,
    *,
    visible_box_hint: bool = True,
):
    """
    Plotly (tuỳ chọn): vùng hover trong suốt + tooltip. Nếu visible_box_hint=False
    thì không vẽ viền/fill màu (chỉ tooltip khi trỏ vào trong box).
    """
    if go is None:
        raise ImportError("plotly is required for build_hover_figure; pip install plotly")
    rgb = _load_rgb(image_path)
    h, w = rgb.shape[:2]
    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb))

    if label_path and label_path.is_file():
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        boxes = yolo_lines_to_boxes(lines, w, h) if lines else []

        for cls_id, x1, y1, x2, y2 in boxes:
            if class_names and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = f"class_{cls_id}"
            xs = [x1, x2, x2, x1, x1]
            ys = [y1, y1, y2, y2, y1]
            hover = (
                f"<b>{name}</b><br>"
                f"class id: {cls_id}<br>"
                f"x1,y1: {x1:.1f}, {y1:.1f}<br>"
                f"x2,y2: {x2:.1f}, {y2:.1f}<br>"
                f"w×h: {x2 - x1:.1f} × {y2 - y1:.1f} px"
            )
            if visible_box_hint:
                fill_c = "rgba(255, 165, 0, 0.12)"
                line_kw = dict(width=2, color="rgba(255,140,0,0.85)")
            else:
                fill_c = "rgba(0,0,0,0)"
                line_kw = dict(width=0)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    fillcolor=fill_c,
                    line=line_kw,
                    mode="lines",
                    hoveron="fills",
                    hoverinfo="text",
                    hovertext=hover,
                    name=name,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=str(image_path.name),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        dragmode=False,
        hovermode="closest",
    )
    return fig


def _collect_images(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
    return [str(p) for p in paths]


def launch_annotation_explorer(
    dataset_root: Path,
    split: str = "train",
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    class_names = parse_data_yaml_names(yaml_path) if yaml_path.is_file() else None

    img_root = dataset_root / split / "images"
    lbl_root = dataset_root / split / "labels"
    image_paths = _collect_images(img_root)
    if not image_paths:
        raise FileNotFoundError(f"No images under {img_root}")

    choices = [Path(p).name for p in image_paths]
    path_by_name = {Path(p).name: p for p in image_paths}

    def update_view(filename: str) -> str:
        p = Path(path_by_name[filename])
        lbl = lbl_root / (p.stem + ".txt")
        return build_hover_review_html(p, lbl if lbl.is_file() else None, class_names)

    with gr.Blocks(title="Annotation hover") as demo:
        gr.Markdown(
            "### Xem nhãn theo hover\n"
            "**Ảnh không có box** cho đến khi bạn đưa chuột vào **đúng vùng** có annotation "
            "(viền cam + tooltip). Ra ngoài vùng đó thì khung biến mất."
        )
        dropdown = gr.Dropdown(choices=choices, value=choices[0], label="Frame")
        view = gr.HTML(label="Preview")
        dropdown.change(fn=update_view, inputs=[dropdown], outputs=[view])
        demo.load(fn=update_view, inputs=[dropdown], outputs=[view])

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share)


def main() -> None:
    p = argparse.ArgumentParser(description="Gradio UI: hover để hiện box annotation")
    p.add_argument(
        "--dataset",
        type=Path,
        default=OUTPUT_DIR / "auto_annotated",
        help="Folder with data.yaml and train/images, train/labels",
    )
    p.add_argument("--split", default="train", choices=("train", "valid", "test"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    launch_annotation_explorer(
        args.dataset,
        split=args.split,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
