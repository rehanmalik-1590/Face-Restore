# Face Restoration with CodeFormer

This project provides a **face restoration pipeline** using [CodeFormer](https://github.com/sczhou/CodeFormer) with optional background upscaling via [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).  
It restores blurry, low-quality, or old facial images by balancing **fidelity** (identity preservation) and **quality enhancement**.

---

## ✨ Features
- Face detection and alignment (RetinaFace / YOLOv5).
- High-quality **face restoration** with CodeFormer.
- Optional **background upscaling** using Real-ESRGAN.
- Flexible fidelity control (`-w`).
- Works for **single images** (this script is image-focused).

---

## 📦 Installation

   ## 🚀 Step-by-Step Guide

### 1. Clone this Repository

git clone https://github.com/MRizwanMalik/Face-Restore.git

cd Face-Restore
### 2. Create Virtual Env
py -3.10 -m venv venv

venv\Scripts\activate
### 3.Install Dependencies
pip install -r requirements.txt

pip install basicsr facelib gfpgan realesrgan
### 4.Run Face Restoration
python test.py

cmd   python test.py -i input.jpg -o ./results/ -w 0.8 -s 2


## Parameters

-i / --input_path : Path to input image.

-o / --output_path : Directory to save results.

-w / --fidelity_weight : Balance between quality and identity (range 0 → 1).

0 = better quality, may lose some identity.

1 = stronger identity preservation, less quality.

-s / --upscale : Final upscaling factor (1, 2, or 4).

--draw_box : Draw bounding boxes around detected faces.

--only_center_face : Restore only the main face in the image.

## 📚 References

CodeFormer (ECCV 2022)

Real-ESRGAN

GFPGAN

⚠️ Disclaimer: All models belong to their original authors.
