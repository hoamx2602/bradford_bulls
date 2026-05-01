"""
Export annotation package -> CVAT 1.1 video annotation XML.

CVAT XML format reference:
    https://opencv.github.io/cvat/docs/manual/advanced/xml_format/

Each track exports as a <track> element with interpolated bboxes per frame.
This lets the annotator use CVAT's track mode if they want to refine the
auto-generated tracking before brand assignment.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from track_annotation.utils.logging import get_logger

log = get_logger(__name__)


def export_to_cvat(
    package_dir: str | Path,
    output_path: str | Path,
    label_name: str = "logo",
) -> Path:
    """
    Export an annotation package to CVAT XML.

    Parameters
    ----------
    package_dir : str | Path
        Annotation package directory.
    output_path : str | Path
        Output XML file path.
    label_name : str
        Label name to use for all tracks (annotator will refine in CVAT).
    """
    package_dir = Path(package_dir).resolve()
    output_path = Path(output_path).resolve()
    manifest = json.loads((package_dir / "manifest.json").read_text())

    annotations_root = ET.Element("annotations")
    ET.SubElement(annotations_root, "version").text = "1.1"

    meta = ET.SubElement(annotations_root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = manifest["video"]["filename"]
    ET.SubElement(task, "size").text = str(manifest["video"]["frame_count"])
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "5"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "created").text = datetime.now(timezone.utc).isoformat()
    ET.SubElement(task, "updated").text = datetime.now(timezone.utc).isoformat()
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(manifest["video"]["frame_count"] - 1)
    ET.SubElement(task, "frame_filter").text = ""

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = label_name
    ET.SubElement(label, "color").text = "#33ddff"
    ET.SubElement(label, "attributes").text = ""

    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(manifest["video"]["width"])
    ET.SubElement(original_size, "height").text = str(manifest["video"]["height"])

    # Tracks
    for track_dir in sorted((package_dir / "tracks").iterdir()):
        if not track_dir.is_dir():
            continue
        meta_data = json.loads((track_dir / "meta.json").read_text())
        track_el = ET.SubElement(annotations_root, "track", {
            "id": str(meta_data["track_id"]),
            "label": label_name,
            "source": "auto",
        })
        for det in meta_data["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            ET.SubElement(track_el, "box", {
                "frame": str(det["frame_idx"]),
                "outside": "0",
                "occluded": "0",
                "keyframe": "1",
                "xtl": f"{x1:.2f}",
                "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}",
                "ybr": f"{y2:.2f}",
            })
        # Add an "outside" box one frame after to terminate the interpolation
        last_frame = meta_data["detections"][-1]["frame_idx"] + 1
        ET.SubElement(track_el, "box", {
            "frame": str(last_frame),
            "outside": "1",
            "occluded": "0",
            "keyframe": "1",
            "xtl": "0",
            "ytl": "0",
            "xbr": "0",
            "ybr": "0",
        })

    rough = ET.tostring(annotations_root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pretty)
    log.info(f"CVAT XML written: {output_path}")
    return output_path
