from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys  # <-- thêm

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)

def _auto_acc_class_ids(model: YOLO, desired: List[str]) -> List[int]:
    """
    Map tên class mong muốn -> id có trong model (khớp lowercase, ví dụ 'cell phone').
    Nếu tên không tồn tại trong model.names thì bỏ qua.
    """
    names = getattr(getattr(model, "model", model), "names", None)
    if names is None:
        _ = model.predict([[0,0,1,1]], verbose=False)
        names = getattr(getattr(model, "model", model), "names", {})
    id_by_name = {str(v).lower(): k for k, v in names.items()}
    ids = []
    for want in desired:
        k = id_by_name.get(want.lower())
        if k is not None:
            ids.append(int(k))
    return sorted(set(ids))

def detect_and_group(
    image_path: str,
    out_dir: str = "outs",
    yolo_weights: str = "yolo11m.pt",
    person_conf: float = 0.4,
    acc_conf: float = 0.25,
    iou_threshold: float = 0.30,
    pad_ratio: float = 0.05,
    desired_acc_names: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Trả về danh sách mỗi người gồm:
    {
        'pid': int,
        'person_path': str,
        'accessory_paths': List[str],
        'accessories': List[{'path': str, 'name': str, 'class_id': int}],
        'bbox': [x1,y1,x2,y2]
    }
    Đồng thời lưu crop vào out_dir.
    """
    if desired_acc_names is None:
        desired_acc_names = ["backpack","umbrella","handbag","tie",
                             "suitcase","bottle","laptop","cell phone","book"]

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    yolo = YOLO(yolo_weights)
    names_map = getattr(getattr(yolo, "model", yolo), "names", {})  # <-- thêm: map id -> tên

    # Lấy id class phụ kiện theo names của model
    acc_ids = _auto_acc_class_ids(yolo, desired_acc_names)

    # 1) Detect person
    res_person = yolo.predict(image_path, classes=[0], conf=person_conf, verbose=False)[0]
    img = res_person.orig_img
    H, W = img.shape[:2]

    persons = []
    for pb in res_person.boxes:
        x1, y1, x2, y2 = map(int, pb.xyxy[0].tolist())
        # nới crop để giữ ngữ cảnh
        padx = int(pad_ratio * (x2 - x1))
        pady = int(pad_ratio * (y2 - y1))
        x1p, y1p = max(0, x1 - padx), max(0, y1 - pady)
        x2p, y2p = min(W, x2 + padx), min(H, y2 + pady)
        crop = img[y1p:y2p, x1p:x2p]
        pid = len(persons)
        p_path = out_dir / f"person_{pid}.jpg"
        cv2.imwrite(str(p_path), crop)
        persons.append({
            "pid": pid,
            "bbox": [x1p, y1p, x2p, y2p],
            "person_path": str(p_path),
            "accessory_paths": [],   # giữ tương thích
            "accessories": []        # <-- mới: chứa {path, name, class_id}
        })

    if not persons:
        return []

    # 2) Detect accessories (nếu có acc_ids)
    if acc_ids:
        res_acc = yolo.predict(image_path, classes=acc_ids, conf=acc_conf, verbose=False)[0]
        # Gán mỗi phụ kiện vào người có IoU lớn nhất
        for bb, cls_id in zip(res_acc.boxes.xyxy, res_acc.boxes.cls):  # <-- lấy cả cls
            a = tuple(map(int, bb.tolist()))
            best_pid, best_iou = -1, 0.0
            for p in persons:
                i = _iou(tuple(p["bbox"]), a)
                if i > best_iou:
                    best_pid, best_iou = p["pid"], i
            if best_pid >= 0 and best_iou >= iou_threshold:
                x1, y1, x2, y2 = a
                roi = img[y1:y2, x1:x2]
                acc_path = out_dir / f"person_{best_pid}_acc_{len(persons[best_pid]['accessory_paths'])}.jpg"
                cv2.imwrite(str(acc_path), roi)

                cls_int = int(cls_id.item())
                acc_name = names_map.get(cls_int, str(cls_int))

                # giữ paths cũ
                persons[best_pid]["accessory_paths"].append(str(acc_path))
                # thêm đối tượng có tên/class
                persons[best_pid]["accessories"].append({
                    "path": str(acc_path),
                    "name": acc_name,
                    "class_id": cls_int
                })

    return persons

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Test YOLO crop & group accessories per person")
    ap.add_argument("--image", required=False, help="Path to input image")
    ap.add_argument("--yolo-weights", default="yolo11m.pt", help="YOLO weights path")
    ap.add_argument("--out-dir", default="outs_image", help="Folder to save crops")
    ap.add_argument("--person-conf", type=float, default=0.4, help="Confidence for person detection")
    ap.add_argument("--acc-conf", type=float, default=0.15, help="Confidence for accessories detection")
    ap.add_argument("--iou-threshold", type=float, default=0.10, help="IoU threshold to assign accessory to person")
    ap.add_argument("--pad-ratio", type=float, default=0.12, help="Padding ratio around person crop")
    ap.add_argument(
        "--acc-names",
        default="backpack,umbrella,handbag,tie,suitcase,bottle,laptop,cell phone,book",
        help="Comma-separated accessory class names to detect (must exist in model.names)"
    )
    ap.add_argument(
        "--list-classes",
        action="store_true",
        help="Print model.names from the YOLO weights and exit"
    )
    args = ap.parse_args()

    if args.list_classes:
        y = YOLO(args.yolo_weights)
        names = getattr(getattr(y, "model", y), "names", {})
        print(json.dumps(names, ensure_ascii=False, indent=2))
        sys.exit(0)

    if not args.image:
        ap.error("--image is required (or use --list-classes).")

    desired = [s.strip() for s in args.acc_names.split(",")] if args.acc_names else None

    results = detect_and_group(
        image_path=args.image,
        out_dir=args.out_dir,
        yolo_weights=args.yolo_weights,
        person_conf=args.person_conf,
        acc_conf=args.acc_conf,
        iou_threshold=args.iou_threshold,
        pad_ratio=args.pad_ratio,
        desired_acc_names=desired
    )

    print(json.dumps({"image": args.image, "persons": results}, ensure_ascii=False, indent=2))
    print(f"[OK] Saved crops to: {args.out_dir}")
