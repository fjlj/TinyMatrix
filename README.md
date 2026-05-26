TinyMatrix
==========

> A lightweight, zero-dependency, 16-bit half-float (fp16) and integer matrix engine, built from scratch for low-level machine learning and systems math.

Overview
--------

TinyMatrix was originally born in 2018 as a sandbox to experiment with custom matrix logic. It has since been completely overhauled into a memory-safe, mathematically rigid compute engine.

Instead of relying on heavy external libraries like BLAS or Eigen, TinyMatrix uses a custom virtual-machine-style dispatch router and bitwise memory unions (mFloat and mHalf) to manipulate data at the lowest levels of C++. It natively supports deep learning operations, making it a perfect drop-in library for lightweight software neural networks.

Key Features
------------

*   **Dual-Type Architecture:** Seamlessly switches between int16\_t for high-speed integer math and custom IEEE-754 half-precision floats (mHalf).
    
*   **Strict Memory Safety:** Modernized with std::initializer\_list to prevent stack reads, and meticulously crafted memory boundaries to eliminate pointer-aliasing during complex operations like A.dot(A, B).
    
*   **Neural Network Ready:** Natively supports Forward and Backward propagation with built-in opcodes for **ReLU**, **Sigmoid**, their respective **Derivatives**, and the **Hadamard Product** (Element-wise multiplication).
    
*   **IEEE-754 Bulletproof:** Custom bit-shifting translation layer safely traps and converts extreme floating-point states like 0.0f, Infinity, and Quiet NaNs without triggering undefined compiler behavior.
    
*   **Zero Dependencies:** Pure C++. No external math libraries required.
    

Quick Start
-----------

### 1\. Initialization & Standard Math

TinyMatrix uses a clean initialization syntax and intelligently handles type promotion. If you multiply an integer matrix by a float, it safely upgrades the matrix memory state.

C++

```c++
#include "TinyMatrix.h"

// Initialize with modern C++ initializer lists
TinyMatrix IntMat(2, 2, { 1, 2, 3, 4 });
TinyMatrix FloatMat(2, 2, { 1.5, -2.5, 3.14, 4.0 });

// Operations can be elegantly chained
IntMat.add(5).multiply(2);

// Dot Product with auto-reshaping
TinyMatrix Output;
Output.dot(IntMat, FloatMat);
```

### 2\. Neural Network Operations

The engine is purpose-built to act as a computational graph for machine learning. You can execute an entire forward or backward pass using chained matrix operations.

C++

```c++
TinyMatrix Inputs(1, 3, { 0.5, -1.2, 3.3 });
TinyMatrix Weights(3, 3, { ... });
TinyMatrix Biases(1, 3, { ... });

// --- Forward Propagation ---  //
Output = (Inputs • Weights) + Biases -> Sigmoid Activation  TinyMatrix Output;
Output.dot(Inputs, Weights).add(Biases).Sigmoid();

// --- Backpropagation (Delta Calculation) ---  //
TinyMatrix Error = Output;
Error.sub(Target);
TinyMatrix Derivative = Output;
Derivative.D_Sigmoid();  // Calculate gradient curve
TinyMatrix Delta = Error;
Delta.hadamard(Derivative); // Element-wise multiplication
```

### 3\. Edge-Case Safety

TinyMatrix handles extreme edge cases gracefully. You can reshape arrays on the fly to flatten 2D image grids into 1D data streams, and it perfectly maintains memory integrity when bombarded with CPU float extremes.

C++

```c++
TinyMatrix ExtremeMat(2, 2, {std::numeric_limits::infinity(), std::numeric_limits::quiet_NaN(), 0.0f, 65504.0f });
```

Testing
-------

TinyMatrix includes a comprehensive test suite (TinyMatrixDemo.cpp) that aggressively asserts memory bounds, float precision drift, neural network calculus gradients, and IEEE-754 type stripping.

Disclaimer
----------

While this engine is stable and mathematically verified, it was written as an exploration of low-level C++ mechanics, memory management, and computational math. It is an educational tool first and foremost.
