# 🚀 CUDA Kernel Practice Repository

This repository is a collection of **CUDA `.cu` kernel implementations** that I have written while learning and practicing **GPU programming with NVIDIA CUDA**.  
The goal of this project is to strengthen my understanding of **parallel programming concepts**, experiment with **optimization techniques**, and build a personal reference library of CUDA kernels.

---

## 📌 What’s Inside

Each file in this repository contains a standalone kernel (or set of kernels) written to solve a specific computational problem or demonstrate a GPU optimization pattern.  
Some examples include:
- 🔹 **Vector and Matrix Operations** – basic CUDA kernels for addition, multiplication, transposition.  
- 🔹 **Shared Memory Optimization** – using shared memory to reduce global memory access.  
- 🔹 **Tiling and Blocking** – techniques for optimizing matrix multiplication and stencil computations.  
- 🔹 **Warp-Level Programming** – experimenting with shuffle and reduction intrinsics. 
- 🔹 **Memory Coalescing** – reorganizing data accesses to maximize bandwidth.  
- 🔹 **Parallel Reductions & Prefix Sums** – classic GPU-friendly algorithms.  

> The repository is continuously growing as I learn and add new kernels.

---

## 🎯 Purpose

- Practice writing efficient CUDA code.  
- Explore **GPU optimization techniques** (memory hierarchy, warps, occupancy, etc.).  
- Build a portfolio of work that demonstrates my progress in **high-performance GPU programming**.  
- Provide a reference for myself (and others learning CUDA).  

---

## ⚙️ How to Run

1. Make sure you have:
   - NVIDIA GPU with CUDA support  
   - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed  
   - A C++ compiler (e.g., `g++`)  

2. Compile any kernel file with `nvcc`:
   ```bash
   nvcc my_kernel.cu -o my_kernel
