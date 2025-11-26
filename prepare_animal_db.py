import os
import random
import shutil
from pathlib import Path

# Ruta del dataset 
SOURCE_ROOT = Path(r"C:\Users\1108361294\OneDrive\Documentos\VA_Final\raw-img")

# Carpeta destino
TARGET_ROOT = Path("animal_db")

# Máximo de imágenes por clase a copiar
SAMPLES_PER_CLASS = 90


def main():
    print("[DEBUG] Buscando subcarpetas dentro de:", SOURCE_ROOT)

    if not SOURCE_ROOT.is_dir():
        print("[ERROR] SOURCE_ROOT no es una carpeta válida. Revisa la ruta.")
        return

    TARGET_ROOT.mkdir(exist_ok=True)

    class_dirs = [d for d in SOURCE_ROOT.iterdir() if d.is_dir()]
    if not class_dirs:
        print("[WARN] No se encontraron subcarpetas dentro de:", SOURCE_ROOT)
        return

    print("[INFO] Clases encontradas en el dataset:")
    for d in class_dirs:
        print("   -", d.name)

    for src_dir in class_dirs:
        src_name = src_dir.name
        dst_name = src_name.lower()  
        dst_dir = TARGET_ROOT / dst_name
        dst_dir.mkdir(exist_ok=True, parents=True)

        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not images:
            print(f"[WARN] Sin imágenes en {src_dir}")
            continue

        random.shuffle(images)
        selected = images[:SAMPLES_PER_CLASS]

        print(f"[INFO] Copiando {len(selected)} imágenes de '{src_name}' -> '{dst_name}'")

        for fname in selected:
            src_path = src_dir / fname
            dst_path = dst_dir / fname
            shutil.copy2(src_path, dst_path)

    print("[OK] animal_db preparado.")


if __name__ == "__main__":
    main()
