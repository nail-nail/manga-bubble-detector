"""Grow each speech/thought box out to the balloon outline around it.

Classical image processing only (no model): mark ink with an adaptive threshold, find the white
region that fills the labelled box, drop thin parts like the tail, and take its bounding box plus
the outline stroke. A box is never shrunk. When the white region leaks past the balloon (no closed
outline), the box is left as it is and flagged for a manual check.

Used once on the One Piece external set, whose speech boxes were first drawn around the text. The
proposals were checked by eye before being written back to Label Studio.

Usage:
    python -m src.snap_labels <export.json> <images dir> <out dir> [page numbers, e.g. 013,028]
Writes snap_results.json and one overlay per page: blue = as drawn, green = snapped, red = flagged.
"""
import json, sys, cv2, numpy as np
from pathlib import Path

MAX_GROWTH = 3.5   # snapped area / original area above this counts as a leak

def _largest_overlap(n, roi):
    counts = np.bincount(roi[roi > 0].ravel(), minlength=n)
    return counts.argmax() if counts.sum() else 0

def snap_page(gray, boxes):
    H, W = gray.shape
    ink = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 10)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))  # close one-pixel gaps in the outline
    n, labels = cv2.connectedComponents((ink == 0).astype(np.uint8), connectivity=4)
    out = []
    for x, y, w, h in boxes:
        m = max(6, int(0.08 * min(w, h)))   # look a little outside the text: the balloon surrounds it
        rx0, ry0 = max(0, int(x) - m), max(0, int(y) - m)
        rx1, ry1 = min(W, int(np.ceil(x + w)) + m), min(H, int(np.ceil(y + h)) + m)
        label = _largest_overlap(n, labels[ry0:ry1, rx0:rx1])
        if not label:
            out.append(("fail", (x, y, w, h), "no white")); continue
        # Work in a window around the box; anything that reaches past it is a leak anyway.
        wx0, wy0 = max(0, int(x - 1.5 * w)), max(0, int(y - 1.5 * h))
        wx1, wy1 = min(W, int(x + 2.5 * w)), min(H, int(y + 2.5 * h))
        comp = (labels[wy0:wy1, wx0:wx1] == label).astype(np.uint8)
        # Fill the holes the text makes, so the balloon is one solid blob.
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comp, contours, -1, 1, thickness=cv2.FILLED)
        # Joined balloons share one white region: give each pixel to the nearest text box.
        gy, gx = np.mgrid[wy0:wy1, wx0:wx1]
        def rect_dist(b):
            dx = np.maximum(np.maximum(b[0] - gx, 0), gx - (b[0] + b[2]))
            dy = np.maximum(np.maximum(b[1] - gy, 0), gy - (b[1] + b[3]))
            return np.hypot(dx, dy)
        own = rect_dist((x, y, w, h))
        for other in boxes:
            if other != (x, y, w, h):
                comp[rect_dist(other) < own] = 0
        # Opening removes thin parts: the tail, and narrow leaks into a neighbouring balloon.
        k = max(9, int(0.3 * min(w, h)) | 1)
        comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n2, lab2 = cv2.connectedComponents(comp, connectivity=4)
        best = _largest_overlap(n2, lab2[ry0 - wy0:ry1 - wy0, rx0 - wx0:rx1 - wx0])
        if not best:
            out.append(("fail", (x, y, w, h), "vanished")); continue
        ys, xs = np.nonzero(lab2 == best)
        touches_window = xs.min() == 0 or ys.min() == 0 or xs.max() == comp.shape[1] - 1 or ys.max() == comp.shape[0] - 1
        # A balloon fills most of its bounding box; a leak into the art is ragged and sparse.
        fill = len(xs) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        pad = 4  # the outline stroke itself
        bx0, by0 = max(0, wx0 + xs.min() - pad), max(0, wy0 + ys.min() - pad)
        bx1, by1 = min(W, wx0 + xs.max() + 1 + pad), min(H, wy0 + ys.max() + 1 + pad)
        nx0, ny0, nx1, ny1 = min(bx0, x), min(by0, y), max(bx1, x + w), max(by1, y + h)
        growth = (nx1 - nx0) * (ny1 - ny0) / (w * h)
        if fill < 0.6:
            out.append(("fail", (x, y, w, h), f"sparse {fill:.2f}")); continue
        if growth > MAX_GROWTH or (touches_window and not (wx0 == 0 or wy0 == 0 or wx1 == W or wy1 == H)):
            out.append(("fail", (x, y, w, h), f"leak x{growth:.1f}")); continue
        out.append(("ok", (nx0, ny0, nx1 - nx0, ny1 - ny0), f"x{growth:.2f} fill {fill:.2f}"))
    return out

if __name__ == "__main__":
    export, images, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    pages = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(export).read_text(encoding="utf-8"))
    results = {}
    for t in tasks:
        if not t["annotations"]: continue
        name = t["file_upload"].split("-", 1)[1]
        if pages and name[3:6] not in pages: continue
        gray = cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        regs = [r for r in t["annotations"][-1]["result"]
                if r["value"]["rectanglelabels"][0] in ("speech", "thought")]
        boxes = [(r["value"]["x"]/100*W, r["value"]["y"]/100*H, r["value"]["width"]/100*W, r["value"]["height"]/100*H) for r in regs]
        snapped = snap_page(gray, boxes)
        results[name] = [{"id": r["id"], "status": s, "box": b, "note": note, "orig": o}
                         for r, (s, b, note), o in zip(regs, snapped, boxes)]
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (s, b, note), o in zip(snapped, boxes):
            cv2.rectangle(canvas, (int(o[0]), int(o[1])), (int(o[0]+o[2]), int(o[1]+o[3])), (255, 120, 0), 2)
            color = (0, 170, 0) if s == "ok" else (0, 0, 255)
            cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), color, 4)
        s = 1100 / H
        cv2.imwrite(str(outdir / f"{name[:-4]}.jpg"), cv2.resize(canvas, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 82])
    (outdir / "snap_results.json").write_text(json.dumps(results, indent=1, default=float))
    ok = sum(r["status"] == "ok" for v in results.values() for r in v); tot = sum(len(v) for v in results.values())
    print(f"{ok}/{tot} snapped; failed:", [(k[3:6], r["id"][:5], r["note"]) for k, v in results.items() for r in v if r["status"] != "ok"])
