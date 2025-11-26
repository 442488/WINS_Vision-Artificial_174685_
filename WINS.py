import os
import time
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Parametros

CONF_THRESH = 0.32
IMG_SIZE    = 256
CAP_W, CAP_H = 256, 192
DISP_W, DISP_H = 640, 480
PROCESS_EVERY_N = 3
MAX_DET     = 10
IOU_THRESH  = 0.5

KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
LOWER_BAND_FRAC  = 0.55
MEDIUM_BAND_FRAC = 0.35

ANIMAL_DB_ROOT = "animal_db"
ORB_FEATURES   = 500

# Clases para YOLO-World
PROMPT_CLASSES = [
    "tlacuache","opossum","possum","didelphis virginiana",
    "cacomixtle","ringtail","ring-tailed cat","bassariscus astutus",
    "mammal","animal",
    "cat","dog","horse","cow","sheep","goat",
    "bird","pigeon","sparrow","chicken",
    "squirrel","rabbit","butterfly","spider","elephant",
    "person","cell phone"
]

# Funciones de UI 

def recommended_action(label: str) -> str:
    l = label.lower()
    if any(k in l for k in ["tlacuache","zarigueya","zarigüeya","opossum","possum",
                            "didelphis","cacomixtle","ringtail","bassariscus"]):
        return "ACTION: Reduce speed; block access; guide wildlife to safe exit."
    if any(k in l for k in ["perro","dog"]):
        return "ACTION: Slow down; avoid sudden maneuvers; notify animal services if needed."
    if any(k in l for k in ["gato","cat"]):
        return "ACTION: Slow down; avoid sudden maneuvers; check surroundings."
    if any(k in l for k in ["vaca","cow","oveja","sheep","goat","cabrito"]):
        return "ACTION: Stop; allow crossing; warn other drivers."
    if any(k in l for k in ["caballo","horse"]):
        return "ACTION: Reduce speed; keep distance from horse."
    if any(k in l for k in ["elefante","elephant"]):
        return "ACTION: Maintain long distance; do not block animal path."
    if any(k in l for k in ["ardilla","squirrel","conejo","rabbit"]):
        return "ACTION: Slow down; let the animal pass safely."
    if any(k in l for k in ["gallina","bird","chicken","pigeon","sparrow","ave","pájaro","pajaro"]):
        return "ACTION: Avoid noise; keep distance; check overhead space."
    if any(k in l for k in ["mariposa","butterfly"]):
        return "ACTION: No action required; monitor only."
    if any(k in l for k in ["araña","arana","spider"]):
        return "ACTION: No traffic action; avoid direct contact."
    if "person" in l or "persona" in l:
        return "ACTION: Pedestrian detected; keep safe distance and low speed."
    if any(k in l for k in ["mammal","animal","wildlife"]):
        return "ACTION: Check surroundings; keep safe distance."
    return "ACTION: Observe surroundings; keep safe distance."

def draw_label(img, text, x, y, color=(0, 0, 0)):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 3, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_banner_top(img, text):
    H, W = img.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    cv2.rectangle(img, (0, 0), (W, th + pad * 2), (20, 20, 20), -1)
    cv2.putText(img, text, (pad, th + pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_banner_bottom(img, text):
    H, W = img.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    y1 = H - th - pad * 2
    cv2.rectangle(img, (0, y1), (W, H), (40, 40, 40), -1)
    cv2.putText(img, text, (pad, H - pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# Visión clasica

def classical_preprocess(frame_bgr):
    frame_resized = cv2.resize(frame_bgr, (CAP_W, CAP_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return frame_resized, gray_blur, mag_norm

def classical_motion_mask(bg_subtractor, gray_blur):
    fg = bg_subtractor.apply(gray_blur)
    _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, KERNEL_OPEN, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=2)
    return closed

def estimate_risk_from_yolo(detections, frame_shape):
    if not detections:
        return "LOW"
    H, W = frame_shape[:2]
    high = False
    medium = False
    for x1, y1, x2, y2, conf, name in detections:
        w = x2 - x1
        h = y2 - y1
        area_frac = (w * h) / float(W * H)
        cy = y2 - h / 2.0
        if area_frac > 0.05 and cy > H * LOWER_BAND_FRAC:
            high = True
            break
        if area_frac > 0.01 and cy > H * MEDIUM_BAND_FRAC:
            medium = True
    if high:
        return "HIGH"
    if medium:
        return "MEDIUM"
    return "LOW"

# Base de datos ORB 

def load_animal_db(db_root):
    db = {}
    if not os.path.isdir(db_root):
        print(f"[WARN] Animal DB folder '{db_root}' not found.")
        return db, None

    orb = cv2.ORB_create(ORB_FEATURES)

    for label in os.listdir(db_root):
        class_dir = os.path.join(db_root, label)
        if not os.path.isdir(class_dir):
            continue
        desc_list = []
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path = os.path.join(class_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, des = orb.detectAndCompute(gray, None)
            if des is not None:
                desc_list.append(des)
        if desc_list:
            db[label] = desc_list
            print(f"[DB] {label}: {len(desc_list)} samples")
    if not db:
        print("[WARN] Animal DB is empty.")
    return db, orb

def classify_patch_with_db(patch_bgr, db, orb, distance_thresh=60.0):
    if not db or orb is None:
        return None, None
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    if des is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_label = None
    best_score = 0.0
    for label, desc_list in db.items():
        total_matches = 0
        total_good = 0
        for db_des in desc_list:
            matches = bf.match(des, db_des)
            if not matches:
                continue
            matches = sorted(matches, key=lambda m: m.distance)
            good = [m for m in matches if m.distance < distance_thresh]
            total_matches += len(matches)
            total_good += len(good)
        if total_matches == 0:
            continue
        score = total_good / float(total_matches)
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None or best_score < 0.05:
        return None, None
    return best_label, best_score

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--show", type=lambda v: str(v).lower() in ["1","true","yes","y"],
                        default=True)
    parser.add_argument("--mirror", type=lambda v: str(v).lower() in ["1","true","yes","y"],
                        default=False)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--debug_cv", type=lambda v: str(v).lower() in ["1","true","yes","y"],
                        default=False)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", "GPU" if device == 0 else "CPU")

    # Modelo YOLOv8
    try:
        model = YOLO("yolov8s-world.pt")
        try:
            model.set_classes(PROMPT_CLASSES)
            print("Using YOLO-World with prompts.")
        except Exception:
            print("YOLO-World loaded, but set_classes not supported.")
    except Exception as e:
        print("Could not load yolov8s-world.pt:", e, "=> using yolov8n.pt.")
        model = YOLO("yolov8n.pt")

    _ = model.predict(
        source=np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8),
        imgsz=IMG_SIZE, conf=CONF_THRESH, iou=IOU_THRESH, max_det=1,
        device=device, verbose=False
    )

    animal_db, orb = load_animal_db(ANIMAL_DB_ROOT)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 30, (DISP_W, DISP_H))
        if not writer.isOpened():
            print("Could not open output file.")
            writer = None
        else:
            print(f"Recording to: {args.save}")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=True
    )

    print("Streaming… (press Q to quit)")
    t0 = time.time()
    frames = 0
    last_dets = []
    last_action = "ACTION: Normal monitoring."
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok:
                print("Frame not available.")
                break

            frames += 1
            frame_infer, gray_blur, sobel_mag = classical_preprocess(frame_raw)
            motion_mask = classical_motion_mask(bg_subtractor, gray_blur)

            run_yolo = (frames % PROCESS_EVERY_N == 0)
            if run_yolo:
                results = model.predict(
                    source=frame_infer,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    max_det=MAX_DET,
                    device=device,
                    agnostic_nms=True,
                    verbose=False
                )
                dets = []
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    r = results[0]
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                    conf = r.boxes.conf.cpu().numpy()
                    cls  = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names
                    H, W = frame_infer.shape[:2]

                    for i_det in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i_det].tolist()
                        c = float(conf[i_det])
                        cls_id = int(cls[i_det])

                        if isinstance(names, dict):
                            name_yolo = names.get(cls_id, f"class_{cls_id}")
                        else:
                            name_yolo = str(names[cls_id])

                        x1c = max(0, min(W - 1, x1))
                        y1c = max(0, min(H - 1, y1))
                        x2c = max(0, min(W, x2))
                        y2c = max(0, min(H, y2))
                        if x2c <= x1c or y2c <= y1c:
                            continue
                        patch = frame_infer[y1c:y2c, x1c:x2c]

                        label_db, _ = classify_patch_with_db(patch, animal_db, orb)
                        label_used = label_db if label_db is not None else name_yolo

                        dets.append((x1, y1, x2, y2, c, label_used))

                    if dets:
                        last_action = recommended_action(dets[0][5])

                last_dets = dets

            frame_disp = cv2.resize(frame_infer, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
            sx = DISP_W / CAP_W
            sy = DISP_H / CAP_H

            for (x1, y1, x2, y2, c, label_used) in last_dets:
                X1, Y1 = int(x1 * sx), int(y1 * sy)
                X2, Y2 = int(x2 * sx), int(y2 * sy)
                cv2.rectangle(frame_disp, (X1, Y1), (X2, Y2), (255, 105, 180), 2)
                label_text = f"{label_used} {c:.2f}"
                draw_label(frame_disp, label_text, X1, Y1, color=(255, 105, 180))

            risk = estimate_risk_from_yolo(last_dets, frame_infer.shape)
            risk_counts[risk] += 1

            elapsed = time.time() - t0
            fps = frames / max(1e-6, elapsed)
            top_text = f"Risk: {risk} | ~{fps:.1f} FPS | YOLO + CV + DB | 1/{PROCESS_EVERY_N} YOLO"
            draw_banner_top(frame_disp, top_text)
            draw_banner_bottom(frame_disp, last_action)

            if args.mirror:
                frame_disp = cv2.flip(frame_disp, 1)

            if args.debug_cv:
                cv2.imshow("Sobel magnitude", sobel_mag)
                cv2.imshow("Motion mask", motion_mask)

            if args.show:
                cv2.imshow("WINS", frame_disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

            if writer is not None:
                writer.write(frame_disp)

    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        total = max(1, frames)
        print("\n=== Risk statistics ===")
        for k in ["LOW","MEDIUM","HIGH"]:
            c = risk_counts[k]
            print(f"{k:6s}: {c:5d} frames ({100.0 * c / total:5.1f}%)")
        print("Camera stopped.")

if __name__ == "__main__":
    main()
