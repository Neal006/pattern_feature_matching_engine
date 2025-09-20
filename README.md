# Jersey Pattern Matcher – Simple Step‑by‑Step Guide (Windows)

This guide explains in very simple language how to download, set up, and run your Jersey Pattern Matcher project on a Windows computer. It also explains how to keep your large catalogue (15,000 images) fast by using an NVIDIA GPU with CUDA, and what to do when you add or change catalogue images.

Important files and folders in this project:
- `app.py` – starts the dashboard (Streamlit app).
- `feature_extract.py` – builds the search index from your catalogue images.
- `index/` – stores the index files created by `feature_extract.py`:
  - `index/vector.index`
  - `index/vector.index.paths.txt`
- `catalogue101/` – put your 15,000+ catalogue images here.
- `uploads101/` – every image you upload in the dashboard gets saved here.
- `models/` – contains model files (e.g., `models/deepfashion2_yolov8s-seg.pt` for YOLO).
- `requirement.txt` – list of Python packages to install.

Tip: All paths above are relative to your project folder.

---

## 1) What you need before you start

- A Windows PC. For best speed with large catalogues (15,000 images), an NVIDIA GPU is recommended.
- Admin rights on your PC so you can install software.

Install these:
1. Git
   - Download from: https://git-scm.com/download/win
   - Click through the installer with default options.

2. Python 3.10+ (64-bit)
   - Download from: https://www.python.org/downloads/windows/
   - During install: check “Add Python to PATH”.

3. NVIDIA GPU (optional but strongly recommended for 15,000 images)
   - Make sure you have an NVIDIA graphics card.
   - Install the latest NVIDIA GPU Driver:
     - https://www.nvidia.com/Download/index.aspx

4. PyTorch with CUDA (GPU acceleration)
   - If you have an NVIDIA GPU, you need the CUDA version of PyTorch.
   - We will install this in Step 4 below.
   - If you don’t have a GPU, you can still run on CPU, but it will be slower.

---

## 2) Download the project from GitHub

1. Open “Windows PowerShell”.
2. Choose a folder (like `Documents`) where you want the project to live.
3. Run:
```powershell
git clone <YOUR_GITHUB_REPO_URL>
```
Replace `<YOUR_GITHUB_REPO_URL>` with your repository URL.

4. Go into the project folder (in PowerShell):
```powershell
ls
# Find your cloned folder name, then:
# cd .\pattern_feature_matching_engine\
```
Note: In these instructions, we’ll refer to this folder as your “project folder”.

---

## 3) Create and activate a Python virtual environment

1. Create a virtual environment named `venv_dino`:
```powershell
python -m venv venv_dino
```

2. Activate it:
```powershell
.\venv_dino\Scripts\activate
```

You should now see `(venv_dino)` at the start of your PowerShell prompt.

To deactivate later:
```powershell
deactivate
```

---

## 4) Install dependencies (with CUDA if you have an NVIDIA GPU)

Stay inside your activated virtual environment `(venv_dino)`.

- If you have an NVIDIA GPU (recommended for 15,000 images):
  - Install PyTorch with CUDA 12.1:
  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
  - Verify CUDA is detected:
  ```powershell
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  ```
  Should print `CUDA available: True`. If False, see Troubleshooting at the end.

- Install the rest of the project packages:
```powershell
pip install -r requirement.txt
```

Note: The file is named `requirement.txt` (not `requirements.txt`).

---

## 5) Prepare your folders and models

- Create these folders if they don’t exist yet:
  - `catalogue101/` – put all your catalogue images here (JPG or PNG). For 15,000 images, consider organizing them in subfolders if you like (the code can scan subfolders).
  - `index/` – empty to start. This will be filled by `feature_extract.py`.
  - `uploads101/` – the app will create this automatically on first upload, but you can create it manually too.
  - `models/` – make sure your YOLO model file is present:
    - `models/deepfashion2_yolov8s-seg.pt`
    - If you don’t have it, place the correct YOLO weight file there.

The DINO model is downloaded automatically by the Transformers library the first time it runs.

---

## 6) Build the catalogue index (very important)

You must create the index before running the dashboard. This uses `feature_extract.py` to extract features for each catalogue image and store them in `index/`.

- Run:
```powershell
(venv_dino) python feature_extract.py
```

What happens:
- The script scans `catalogue101/` for images.
- It extracts features using DINO (and may use the GPU if available).
- It saves the index files to `index/vector.index` and `index/vector.index.paths.txt`.

For 15,000 images, this can take some time (first run). With a GPU it will be much faster than CPU.

You only need to rebuild the index when your catalogue changes (see Step 9).

---

## 7) Start the dashboard

Once the index files exist in `index/`, start the Streamlit app:
```powershell
(venv_dino) python -m streamlit run app.py
```

- Streamlit will print a local URL like:
  - Local URL: http://localhost:8501
- Click the link or copy it into your browser.

---

## 8) Using the dashboard

On the dashboard:
- Click “Upload Your Jersey Image.”
- Select a clear image of a jersey (JPG/PNG).
- The app will:
  - Save your uploaded image to `uploads101/` (one copy per unique image).
  - Detect and crop the jersey area using YOLO.
  - Extract features using DINO (and some additional texture/color logic if enabled).
  - Search for similar patterns in your `index/`.
  - Show the top 15 matches with images and scores.
  - Provide a “Download Results” button to save the comparison panel.

Tip:
- If the app says it’s using “Basic DINO Features (384D),” that matches your current index dimension. This is expected unless you rebuild the index with a different feature set.

---

## 9) When you update your catalogue (add/remove images)

Any time you change images in `catalogue101/`:
1. Stop the app if it’s running (press Ctrl+C in the PowerShell window running Streamlit).
2. Rebuild the index:
   ```powershell
   (venv_dino) python feature_extract.py
   ```
   This updates `index/vector.index` and `index/vector.index.paths.txt` so the search reflects your latest catalogue.

3. Start the app again:
   ```powershell
   (venv_dino) python -m streamlit run app.py
   ```

---

## 10) Performance advice for 15,000 images

- Use an NVIDIA GPU and CUDA-enabled PyTorch (Step 4).
- Close other heavy apps while building the index.
- The first time you run the app, models may download and initialize; later runs are faster.
- Keep your images reasonably sized (standard JPG/PNG; extremely large images slow things down).

---

## 11) Folder structure (what it should look like)

Example:
```
your-project-folder/
  app.py
  feature_extract.py
  requirement.txt
  models/
    deepfashion2_yolov8s-seg.pt
  catalogue101/
    <your 15,000 images in subfolders or flat>
  index/
    vector.index
    vector.index.paths.txt
  uploads101/
    <images you uploaded via the dashboard>
```

---

## 12) Troubleshooting (simple fixes)

- The app runs but shows errors about “index”:
  - Make sure you ran `python feature_extract.py` and that `index/vector.index` and `index/vector.index.paths.txt` exist.

- It says the YOLO model file is missing:
  - Put `deepfashion2_yolov8s-seg.pt` in `models/` and run again.

- It’s very slow:
  - Use a GPU with CUDA (Step 4).
  - Confirm CUDA is detected:
    ```powershell
    (venv_dino) python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    ```
  - If it says `False`:
    - Update NVIDIA drivers.
    - Reinstall CUDA PyTorch wheel (Step 4).
    - Try the official guide: https://pytorch.org/get-started/locally/

- “Module not found” errors:
  - Make sure your virtual environment is activated and you installed dependencies:
    ```powershell
    .\venv_dino\Scripts\activate
    pip install -r requirement.txt
    ```

- “Permission denied” or “access is denied”:
  - Close files/folders that are open in other apps.
  - Make sure your antivirus is not blocking Python.

---

## 13) Frequently asked questions

- Can I run without a GPU?
  - Yes, but it will be slower, especially when building the index for 15,000 images.

- Do I need to rebuild the index every time?
  - Only when you add/remove/rename images in `catalogue101/`.

- Where are my uploaded images saved?
  - `uploads101/` inside the project folder. The app avoids saving duplicates of the same image.

- Can I organize the catalogue in subfolders?
  - Yes. The feature extractor scans subfolders too.

---

## 14) Quick command summary

After cloning the repo:
```powershell
python -m venv venv_dino
.\venv_dino\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # for NVIDIA GPU users
pip install -r requirement.txt
python feature_extract.py
python -m streamlit run app.py
```

When catalogue changes:
```powershell
(venv_dino) python feature_extract.py
(venv_dino) python -m streamlit run app.py
```

---

If you need help, open an issue on GitHub with screenshots of the error and the commands you ran. We’re happy to assist.
