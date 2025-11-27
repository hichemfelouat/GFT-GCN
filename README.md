# GFT-GCN: Privacy-Preserving 3D Face Mesh Recognition with Spectral Diffusion

[![My ArXiv Papers](https://img.shields.io/badge/ArXiv-2511.19958-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.19958)

<p align="center">
  <img src="GFT_GCN.png" alt="GFT_GCN" style="max-width: 50%; height: auto;">
</p>

---

## 🧩 Overview

<p align="justify">
3D face recognition is a powerful biometric modality that captures detailed facial geometry, enabling robustness to variations in lighting, pose, occlusion, and presentation attacks. Despite these advantages, protecting stored biometric templates remains a critical challenge.

We introduce **GFT-GCN**, a privacy-preserving 3D face recognition framework that integrates spectral graph learning with diffusion-based template protection. The method applies the **Graph Fourier Transform (GFT)** and **Graph Convolutional Networks (GCNs)** to extract compact and discriminative spectral features from 3D face meshes. A novel **spectral diffusion mechanism** ensures the resulting templates are **irreversible, renewable, and unlinkable**.

To enhance security, the system adopts a lightweight **client–server architecture**, ensuring that raw biometric data never leaves the client side. Experiments conducted on the **BU-3DFE** and **FaceScape** datasets demonstrate strong recognition accuracy and high resistance to reconstruction attacks. Our results show that GFT-GCN achieves an effective trade-off between privacy and performance, offering a practical solution for secure 3D face authentication.
</p>

---

## 🔧 Getting Started

Follow the steps below to set up the environment and prepare your data.

### **1. Clone the Repository**
```bash
git clone https://github.com/hichemfelouat/GFT-GCN.git
cd GFT-GCN
```

### **2. Create and Activate the Conda Environment**
```bash
conda create --name GFT-GCN python=3.10
conda activate GFT-GCN
```

### **3. Install PyTorch and CUDA**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install conda-forge::pytorch_geometric
```

### **4. Install Additional Dependencies**
```bash
pip install trimesh
pip install scipy
pip install pandas
pip install scikit-learn
pip install pytorch-minimize
```

---

## 🗂️ Dataset Preparation

Prepare your dataset by cropping and cleaning the 3D scans if necessary, then organizing them in the following structure:

```
Dataset/
   Subject_1/
      scan_1.obj
      scan_2.obj
      ...
   Subject_2/
      scan_1.obj
      scan_2.obj
      ...
```

After organizing the dataset, extract features by specifying:

- number of subjects  
- parameter **K**  
- number of features  
- number of pairs (match/mismatch)  

Then run:

```bash
python get_dataset.py
```

---

## 🧠 Training

Once feature extraction is complete, proceed to model training. Specify:

- number of features  
- parameter **K**  
- threshold  
- number of diffusion steps  
- number of clients  
- number of GCN and diffusion training epochs  

Then run:

```bash
python train.py
```

Ensure that the dataset path and hyperparameters are set properly based on your dataset size and application requirements.

---

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{felouat2025gftgcnprivacypreserving3dface,
      title={GFT-GCN: Privacy-Preserving 3D Face Mesh Recognition with Spectral Diffusion}, 
      author={Hichem Felouat and Hanrui Wang and Isao Echizen},
      year={2025},
      eprint={2511.19958},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19958}, 
}
```

---

## 📬 Contact

For questions, feedback, or collaboration requests, feel free to reach out:

📧 **hichemfel@gmail.com**  
📧 **hichemfel@nii.ac.jp**

---
