# Comparative Analysis of Adaptive Filtering and Neural Network-Based Approaches for Audio Signal Denoising
## Signal, Image and Video - University of Trento

### M. Prosperi


This project investigates the denoising of audio signals through the application of adaptive filtering techniques, including Least Mean Squares (LMS) and Normalized Least Mean Squares (NLMS), alongside a neural network-based approach. The study is divided into two main phases to systematically evaluate the performance of these methods.

The first phase involves the use of a synthetic audio signal, where each denoising method is assessed both qualitatively (through visual inspection of the results) and quantitatively using objective metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²). This controlled environment allows for a rigorous comparison of the models’ effectiveness under well-defined conditions.

The second phase extends the analysis to real-world audio signals, utilizing the NOIZEUS Speech Corpus (available at https://ecs.utdallas.edu/loizou/speech/noizeus/), a dataset that provides speech signals with various noise conditions. By applying the denoising techniques to this dataset, the project evaluates the practical applicability and robustness of each method in real-world scenarios.

## How to Run the Project

### Option 1: Run with Google Colab (Recommended)
The fastest way to run the project is using Google Colab.  
Simply upload the provided notebook to Colab, and you can run it without installing any dependencies locally. 

### Option 2: Run Locally
If you prefer to run the project on your local machine, ensure you have the following dependencies installed:

#### Dependencies
- **Python 3.8+** (recommended)
- **Python Libraries**:
  - `os` (built-in module, no need to install)
  - `torch` (PyTorch)
  - `torch.nn` (from PyTorch)
  - `torch.optim` (from PyTorch)
  - `numpy` 
  - `matplotlib`
  - `scikit-learn`
  - `scipy`

#### Installation Instructions
To install the required libraries, you can use the following command:
```bash
pip install torch numpy matplotlib scikit-learn scipy
