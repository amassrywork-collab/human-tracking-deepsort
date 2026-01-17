
---

# 🧪 How to Set Up the Environment (Using Anaconda)

> This guide explains **exactly** how to download, run, and verify the environment setup
> using **Anaconda Prompt on Windows**.

---

## ✅ Step 1: Install Anaconda (If Not Installed)

1. Download Anaconda from:
   👉 [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Install it with **default settings**
3. Restart your computer (recommended)

---

## ✅ Step 2: Download the Setup File

1. Download the file:

   ```
   setup_conda_env.bat
   ```
2. Place it in any folder
   *(recommended: the same folder as the project)*

Example:

```
human-tracking-deepsort/
├── setup_conda_env.bat
├── src/
├── data/
└── README.md
```

---

## ✅ Step 3: Open Anaconda Prompt

1. Open **Start Menu**
2. Search for:

   ```
   Anaconda Prompt
   ```
3. Open it
   (Do NOT use CMD or PowerShell)

---

## ✅ Step 4: Navigate to the File Location

Inside **Anaconda Prompt**, move to the folder where the file exists.

Example:

```bash
cd Desktop\human-tracking-deepsort
```

> 📌 Tip:
> You can also **copy the folder path** from File Explorer and paste it here.

---

## ✅ Step 5: Run the Setup File

Execute the file using:

```bash
setup_conda_env.bat
```

Then press **Enter**.

---

## ⏳ Step 6: Wait for Installation to Finish

* The setup will:

  * Create a conda environment named `cv_dl_lab`
  * Install all required libraries
  * Test the installation automatically

⚠️ This may take **5–10 minutes** depending on your internet speed.

---

## ✅ Step 7: Confirm Successful Installation

At the end, you should see:

```
Environment setup completed successfully!
```

If you see this message → 🎉 **Everything is ready**

---

## 🧪 Step 8: Activate the Environment (Every Time You Work)

Each time before running the project:

```bash
conda activate cv_dl_lab
```

You should see:

```
(cv_dl_lab)
```

---

## ▶️ Step 9: Run the Project

### Webcam:

```bash
python src/main.py --source 0 --show
```

### Video file:

```bash
python src/main.py --source data/input.mp4 --show
```

---

## ❗ Common Mistakes to Avoid

| Mistake                            | Correct Action                    |
| ---------------------------------- | --------------------------------- |
| Using CMD / PowerShell             | Always use **Anaconda Prompt**    |
| Running `.bat` by double-click     | Run it **inside Anaconda Prompt** |
| Forgetting to activate environment | Run `conda activate cv_dl_lab`    |
| Closing prompt during install      | Wait until it finishes            |

---

 If Something Goes Wrong

1. Copy the **full error message**
2. Do NOT guess or reinstall randomly
3. Send the error message for help



## ✅ Summary

✔ One file
✔ One command
✔ One environment
✔ Ready for Deep Learning & Computer Vision work

