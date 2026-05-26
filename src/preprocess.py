
import cv2
import json
import time
import multiprocessing
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR   = "data"           
OUTPUT_DIR = "processed_data" 
IMG_SIZE   = (64, 64)         
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  
CHUNK_SIZE  = 500             

CLASSES = {"awake": 0, "sleepy": 1}   
SPLITS  = ["train", "val", "test"]
EXTS    = {".png", ".jpg", ".jpeg", ".bmp"}

# ================================================================
def xu_ly_batch(batch):
    ket_qua = []
    for duong_dan, nhan in batch:
        anh = cv2.imread(duong_dan, cv2.IMREAD_GRAYSCALE)
        if anh is None:
            continue  

        anh = cv2.resize(anh, IMG_SIZE, interpolation=cv2.INTER_AREA)
        
        
        anh = anh[..., np.newaxis] 

        ket_qua.append((anh, nhan))
    return ket_qua

# ================================================================
def xu_ly_split(split):
    danh_sach = []
    for ten_lop, nhan in CLASSES.items():
        thu_muc = Path(DATA_DIR) / split / ten_lop
        for f in sorted(thu_muc.iterdir()):
            if f.suffix.lower() in EXTS:
                danh_sach.append((str(f), nhan))

    tong = len(danh_sach)
    print(f"\n[{split.upper()}] {tong:,} ảnh | {NUM_WORKERS} workers")

    batches = [danh_sach[i:i + CHUNK_SIZE] for i in range(0, tong, CHUNK_SIZE)]

    tat_ca_anh, tat_ca_nhan = [], []
    da_xong = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(xu_ly_batch, b) for b in batches]

        for fut in as_completed(futures):
            for anh, nhan in fut.result():
                tat_ca_anh.append(anh)
                tat_ca_nhan.append(nhan)
            da_xong += 1

            phan_tram = da_xong / len(batches) * 100
            thanh     = "█" * int(phan_tram / 4) + "░" * (25 - int(phan_tram / 4))
            toc_do    = len(tat_ca_anh) / max(time.time() - t0, 1e-6)
            print(f"\r  [{thanh}] {phan_tram:5.1f}%"
                  f"  {len(tat_ca_anh):>7,}/{tong:,}"
                  f"  {toc_do:,.0f} img/s", end="", flush=True)

    print(f"  ✔ {time.time() - t0:.1f}s")


    images = np.stack(tat_ca_anh).astype(np.uint8)  
    labels = np.array(tat_ca_nhan, dtype=np.int32)     
    return images, labels

# ================================================================
def luu_ket_qua(images, labels, split):
    thu_muc = Path(OUTPUT_DIR)
    thu_muc.mkdir(parents=True, exist_ok=True)

    np.save(thu_muc / f"{split}_images.npy", images)
    np.save(thu_muc / f"{split}_labels.npy", labels)

    dung_luong = (thu_muc / f"{split}_images.npy").stat().st_size / 1024**2
    print(f"  Đã lưu: {split}_images.npy  ({dung_luong:.1f} MB)")
    print(f"  Đã lưu: {split}_labels.npy")

# ================================================================
def main():
    print("=" * 50)
    print("  TIỀN XỬ LÝ DỮ LIỆU – TỐI ƯU HÓA")
    print(f"  img_size    : {IMG_SIZE}")
    print(f"  format      : uint8 (0-255)")
    print("=" * 50)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    config = {
        "img_size": IMG_SIZE, "classes": CLASSES,
        "format": "uint8",
    }
    with open(f"{OUTPUT_DIR}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    t_bat_dau = time.time()
    for split in SPLITS:
        images, labels = xu_ly_split(split)
        luu_ket_qua(images, labels, split)
        del images, labels 

    print(f"\n{'─'*50}")
    print(f"  Tổng thời gian: {(time.time()-t_bat_dau)/60:.1f} phút")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()