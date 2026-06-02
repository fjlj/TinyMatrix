/*
 * ===========================================================================
 * TinyMatrix - A lightweight 16-bit half-float & integer matrix engine
 * ===========================================================================
 * * HOW TO USE THIS LIBRARY:
 * * In EXACTLY ONE C++ file, define TINYMATRIX_IMPLEMENTATION before including
 * this header to create the implementation.
 * * Example:
 * #define TINYMATRIX_IMPLEMENTATION
 * #include "TinyMatrix.h"
 * * ===========================================================================
 */

#ifndef TINY_MATRIX_H
#define TINY_MATRIX_H

#include <assert.h>
#include <iostream>
#include <cmath>
#include <initializer_list>
#include <cstring> 
#include <immintrin.h>


typedef uint16_t HALF;

struct mHalf {
    union {
        HALF uVal;
        struct {
            HALF Frac : 10;
            HALF Exp : 5;
            HALF Sign : 1;
        }parts{0};
    };
    mHalf(const uint16_t& t) {
        this->uVal = t;
    }
};

struct nRet {
    float val;
    nRet(const float& _f) {
        this->val = _f;
    };
    nRet() {
        this->val = 0.0f;
    }
    operator float() {
        return val;
    };
    operator int() {
        return (int)val;
    };
    nRet& operator +=(const float& o) {
        this->val += o;
        return *this;
    };
};

struct mFloat {
    union {
        float fVal;
        struct {
            uint32_t Frac : 23;
            uint32_t Exp : 8;
            uint32_t Sign : 1;
        }parts{0};
    };
    mFloat(const float& t) {
        this->fVal = t;
    }
};

class TinyMatrix {
private:
    int rows;
    int cols;
    int size;
    unsigned char* data;
    bool isFloat = false;

    static thread_local TinyMatrix* _scratch;
    static thread_local TinyMatrix* _operandScratch;
    static thread_local size_t _scratchCapacity;
    static thread_local size_t _operandScratchCapacity;
    static thread_local float* _accumScratch;
    static thread_local size_t _accumCapacity;
    static thread_local int32_t* _accumScratchInt;
    static thread_local size_t _accumIntCapacity;
    static thread_local TinyMatrix* _transposeScratch;
    static thread_local size_t _transposeScratchCapacity; 

public:
    void WriteRaw(std::ostream& out) {
        out.write((char*)&this->isFloat, sizeof(bool));
        out.write((char*)this->data, this->size);
    }

    // Overwrites the raw memory block with bit-perfect accuracy
    void ReadRaw(std::istream& in) {
        in.read((char*)&this->isFloat, sizeof(bool));
        in.read((char*)this->data, this->size);
    }
    int Rows() {
        return this->rows;
    }
    int Cols() {
        return this->cols;
    }
    int Size() {
        return this->size;
    }
    bool IsFloat() {
        return this->isFloat;
    }

    TinyMatrix();
    TinyMatrix(int r, int c);

    // Modern initializer lists replace the old unsafe variadics
    TinyMatrix(int r, int c, std::initializer_list<double> nums);
    TinyMatrix(int r, int c, std::initializer_list<int> nums);

    void init(int r, int c);
    ~TinyMatrix();

    TinyMatrix(const TinyMatrix& source);
    TinyMatrix& operator=(const TinyMatrix& source);

    float operator()(int r, int c);
    TinyMatrix& Shape(int r, int c, bool absolute = false);

    void operator()(int r, int c, const int v);
    void operator()(int r, int c, const float v);
    void operator()(int r, int c, const double v);

    TinyMatrix& add(int s);
    TinyMatrix& sub(int s);
    TinyMatrix& add(float s);
    TinyMatrix& sub(float s);
    TinyMatrix& add(double s);
    TinyMatrix& sub(double s);
    TinyMatrix& add(const TinyMatrix& a, bool reshape = false);
    TinyMatrix& sub(const TinyMatrix& a, bool reshape = false);
    TinyMatrix& multiply(int s);
    TinyMatrix& multiply(float s);
    TinyMatrix& multiply(double s);
    TinyMatrix& Relu();
    TinyMatrix& Sigmoid();
    TinyMatrix& D_Relu();
    TinyMatrix& D_Sigmoid();
    TinyMatrix& hadamard(const TinyMatrix& a);
    TinyMatrix& QuantizeQ88();
    TinyMatrix& DequantizeQ88();
    TinyMatrix& fixed_dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& fixed_hadamard(const TinyMatrix& a);
    TinyMatrix& Tanh();
    TinyMatrix& D_Tanh();
    static int16_t tanh_lut[1024];
    static void InitLUT();
    static void CleanupEngine() {
        delete _scratch;
        _scratch = nullptr;
        delete _operandScratch;
        _operandScratch = nullptr;
        delete[] _accumScratch;      // Added
        _accumScratch = nullptr;     // Added
        delete[] _accumScratchInt;   // Added
        _accumScratchInt = nullptr;  // Added
        delete _transposeScratch;
        _transposeScratch = nullptr;
    }
    TinyMatrix& fixed_tanh();
    void print(std::string extra = "");

    TinyMatrix& transpose();
    TinyMatrix& dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& dot(TinyMatrix& a);
    TinyMatrix& Ints();
    TinyMatrix& Floats();
    TinyMatrix& Randomize(float min = -1.0f, float max = 1.0f);
    nRet sum();
    template <typename Func>
    TinyMatrix& mapInline(Func operation, bool forceFloat = false) {
        bool origFloat = this->isFloat;
        bool targetFloat = origFloat || forceFloat;
        this->isFloat = targetFloat;

        int pos = 0; // Track sequentially
        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->cols; j++) {

                float val = origFloat ? halfToFloat(*(uint16_t*)(this->data + pos))
                    : (float)*(int16_t*)(this->data + pos);

                float result = operation(val, i, j);

                if(targetFloat) {
                    *(uint16_t*)(this->data + pos) = floatToHalf(result);
                } else {
                    *(int16_t*)(this->data + pos) = (int16_t)result;
                }
                pos += 2; // Move forward exactly one 16-bit block
            }
        }
        return *this;
    }

    // For Matrix-to-Matrix element-wise operations
    template <typename Func>
    TinyMatrix& mapInline(TinyMatrix& other, Func operation, bool forceFloat = false) {
        // Halt immediately if shapes don't match and aren't broadcastable
        assert(((this->rows == other.rows && this->cols == other.cols) ||
            (other.rows == 1 && other.cols == this->cols) ||
            (other.cols == 1 && other.rows == this->rows) ||
            (other.rows == 1 && other.cols == 1)) &&
            "Matrix dimensions must match or be compatible for broadcasting!");

        bool origFloat = this->isFloat;
        bool otherFloat = other.IsFloat();
        bool targetFloat = origFloat || otherFloat || forceFloat;
        this->isFloat = targetFloat;

        int pos = 0;
        int other_pos = 0;

        bool same_shape = (this->rows == other.rows && this->cols == other.cols);

        for(int i = 0; i < this->rows; i++) {
            // Lock row index to 0 if we are broadcasting a 1xN row vector
            int other_i = (other.rows == 1) ? 0 : i;

            for(int j = 0; j < this->cols; j++) {

                // Recalculate other_pos ONLY if we are actively broadcasting
                if(!same_shape) {
                    int other_j = (other.cols == 1) ? 0 : j;
                    other_pos = (other_i * other.cols + other_j) * 2;
                }

                float val1 = origFloat ? halfToFloat(*(uint16_t*)(this->data + pos))
                    : (float)*(int16_t*)(this->data + pos);

                float val2 = otherFloat ? halfToFloat(*(uint16_t*)(other.data + other_pos))
                    : (float)*(int16_t*)(other.data + other_pos);

                float result = operation(val1, val2, i, j);

                if(targetFloat) {
                    *(uint16_t*)(this->data + pos) = floatToHalf(result);
                } else {
                    *(int16_t*)(this->data + pos) = (int16_t)result;
                }

                pos += 2;
                if(same_shape) {
                    other_pos += 2; // Fast path for matching matrices
                }
            }
        }
        return *this;
    }
private:
    unsigned char* operator[](const int p);
    static HALF floatToHalf(mFloat i);
    static float halfToFloat(mHalf y);

    // Helpers for cheap, capacity-aware snapshots into the persistent scratch registers.
    void _ensureScratchCapacity(TinyMatrix*& reg, size_t& capacity, int rows, int cols);
    void _snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap);

};
#endif

#ifdef TINYMATRIX_IMPLEMENTATION
thread_local TinyMatrix* TinyMatrix::_scratch = nullptr;
thread_local TinyMatrix* TinyMatrix::_operandScratch = nullptr;
thread_local size_t TinyMatrix::_scratchCapacity = 0;
thread_local size_t TinyMatrix::_operandScratchCapacity = 0;
thread_local float* TinyMatrix::_accumScratch = nullptr;
thread_local size_t TinyMatrix::_accumCapacity = 0;
thread_local int* TinyMatrix::_accumScratchInt = nullptr;
thread_local size_t TinyMatrix::_accumIntCapacity = 0;
thread_local TinyMatrix* TinyMatrix::_transposeScratch = nullptr;
thread_local size_t TinyMatrix::_transposeScratchCapacity = 0;

constexpr auto makeEven(int a) {
    return (a + 1 ^ (~a & 1));
}

// Allocate the 2KB of RAM for the Lookup Table
int16_t TinyMatrix::tanh_lut[1024];

// Run this ONCE at boot to pre-calculate the Q8.8 Tanh curve
void TinyMatrix::InitLUT() {
    for(int i = 0; i < 1024; i++) {
        // Convert Q8.8 integer index to true float, run FPU Tanh, and convert back!
        float float_val = (float)i / 256.0f;
        float t_val = std::tanh(float_val);
        tanh_lut[i] = (int16_t)(t_val * 256.0f);
    }
}

TinyMatrix::TinyMatrix() {
    this->rows = 1; this->cols = 1; this->size = 4;
    this->data = new unsigned char[this->size]();
}

TinyMatrix::TinyMatrix(int r, int c) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<double> nums) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
    this->isFloat = true;

    int i = 0;
    for(double val : nums) {
        if(i >= (this->size / 2)) break;
        *(uint16_t*)(*this)[i] = floatToHalf((float)val);
        i++;
    }
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<int> nums) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
    this->isFloat = false;

    int i = 0;
    for(int val : nums) {
        if(i >= (this->size / 2)) break;
        *(int16_t*)(*this)[i] = (int16_t)val;
        i++;
    }
}

void TinyMatrix::init(int r, int c) {
    if(this == nullptr) return;
    delete[] this->data;                 // safe: all constructed objects own a valid buffer
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
}

TinyMatrix::~TinyMatrix() {
    delete[] this->data;
}

TinyMatrix::TinyMatrix(const TinyMatrix& source) {
    this->rows = source.rows;
    this->cols = source.cols;
    this->size = source.size;
    this->data = new unsigned char[this->size]();
    this->isFloat = source.isFloat;
    std::copy(source.data, source.data + source.size, this->data);
}

TinyMatrix& TinyMatrix::operator=(const TinyMatrix& source) {
    if(&source == this) return *this;

    if(source.rows == this->rows && source.cols == this->cols) {
        // Same dimensions: reuse existing buffer (fast path, no allocation).
        // Only copy the payload and type flag.
        this->isFloat = source.isFloat;
        std::copy(source.data, source.data + source.size, this->data);
        return *this;
    }

    // Size change: clean up old memory, then allocate fresh buffer.
    delete[] this->data;
    this->rows = source.rows;
    this->cols = source.cols;
    this->size = source.size;
    this->isFloat = source.isFloat;
    this->data = new unsigned char[this->size]();
    std::copy(source.data, source.data + source.size, this->data);
    return *this;
}

float TinyMatrix::operator()(int r, int c) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Read Out of Bounds!");
    int pos = (r * (this->cols) + c);
    return (this->isFloat ? halfToFloat((uint16_t) * (uint16_t*)((*this)[pos])) : (float)*(int16_t*)((*this)[pos]));
}

TinyMatrix& TinyMatrix::Shape(int r, int c, bool absolute) {
    if(this->rows == r && this->cols == c) return *this;

    int old_r = this->rows;
    int old_c = this->cols;
    int old_size = this->size;

    this->rows = r;
    this->cols = c;
    unsigned char* old_data = &this->data[0];
    this->size = (makeEven((r * c)) * 2);

    // FIX: Fast-path to prevent unnecessary reallocation (and register capacity destruction)
    if(!absolute && old_size == this->size) {
        return *this;
    }

    bool olim = (old_size <= this->size);
    if(absolute) {
        this->data = new unsigned char[this->size]();
        for(int rt = 0; rt < (olim ? old_r : this->rows); rt++) {
            for(int ct = 0; ct < (olim ? old_c : this->cols); ct++) {
                int np = ((rt * this->cols) + ct) * 2;
                int op = ((rt * old_c) + ct) * 2;

                if(np < this->size - 1 && op < old_size - 1) {
                    this->data[np] = old_data[op];
                    this->data[np + 1] = old_data[op + 1];
                } else if(np < this->size - 1) {
                    this->data[np] = 0;
                    this->data[np + 1] = 0;
                }
            }
        }
        delete[] old_data;
    } else {
        this->data = new unsigned char[this->size]();
        std::copy(old_data, old_data + (old_size < this->size ? old_size : this->size), this->data);
        delete[] old_data;
    }
    return *this;
}

void TinyMatrix::operator()(int r, int c, const int v) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Int Write Out of Bounds!");
    if(this->isFloat) {
        (*this)(r, c, (float)v); return;
    }
    int pos = (r * (this->cols) + c);
    *(int16_t*)(*this)[pos] = (int16_t)v;
}

void TinyMatrix::operator()(int r, int c, const float v) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Float Write Out of Bounds!");
    this->isFloat = true;
    int pos = (r * (this->cols) + c);
    *(uint16_t*)(*this)[pos] = floatToHalf(v);
}

void TinyMatrix::operator()(int r, int c, const double v) {
    TinyMatrix::operator()(r, c, (float)v);
}

float TinyMatrix::halfToFloat(mHalf y) {
    __m128i v_half = _mm_cvtsi32_si128(y.uVal);
    __m128 v_float = _mm_cvtph_ps(v_half);
    return _mm_cvtss_f32(v_float);
}

HALF TinyMatrix::floatToHalf(mFloat i) {
    __m128 v_float = _mm_set_ss(i.fVal);
    __m128i v_half = _mm_cvtps_ph(v_float, 0);
    return (HALF)_mm_extract_epi16(v_half, 0);
}

TinyMatrix& TinyMatrix::Ints() {
    if(!this->isFloat) return *this;

    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            float val = (*this)(i, j);
            int pos = (i * this->Cols() + j) * 2;

            // Force the 16-bit integer cast directly into memory
            *(int16_t*)(this->data + pos) = (int16_t)val;
        }
    }
    this->isFloat = false;
    return *this;
}

TinyMatrix& TinyMatrix::add(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val + s; }, promote);
}

TinyMatrix& TinyMatrix::sub(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val - s; }, promote);
}

TinyMatrix& TinyMatrix::multiply(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val * s; }, promote);
}

// Ensure the int/double overrides just forward to the float templates
TinyMatrix& TinyMatrix::add(int s) {
    return this->add((float)s);
}
TinyMatrix& TinyMatrix::sub(int s) {
    return this->sub((float)s);
}
TinyMatrix& TinyMatrix::multiply(int s) {
    return this->multiply((float)s);
}
TinyMatrix& TinyMatrix::add(double s) {
    return this->add((float)s);
}
TinyMatrix& TinyMatrix::sub(double s) {
    return this->sub((float)s);
}
TinyMatrix& TinyMatrix::multiply(double s) {
    return this->multiply((float)s);
}

TinyMatrix& TinyMatrix::add(const TinyMatrix& a, bool reshape) {
    if(reshape) this->Shape(((TinyMatrix&)a).Rows(), ((TinyMatrix&)a).Cols(), true);
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 + v2; });
}

TinyMatrix& TinyMatrix::sub(const TinyMatrix& a, bool reshape) {
    if(reshape) this->Shape(((TinyMatrix&)a).Rows(), ((TinyMatrix&)a).Cols(), true);
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 - v2; });
}

TinyMatrix& TinyMatrix::hadamard(const TinyMatrix& a) {
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 * v2; });
}

TinyMatrix& TinyMatrix::fixed_hadamard(const TinyMatrix& a) {
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) {
        // Q8.8 Fixed-Point math baked directly into the lambda
        int32_t x = (int16_t)v1;
        int32_t y = (int16_t)v2;
        return (float)(int16_t)((x * y) >> 8);
        });
}

TinyMatrix& TinyMatrix::Relu() {
    return this->mapInline([](float val, int r, int c) { return val > 0.0001f ? val : 0.0f; }, true);
}

TinyMatrix& TinyMatrix::D_Relu() {
    return this->mapInline([](float val, int r, int c) { return val > 0.0001f ? 1.0f : 0.0f; }, true);
}

TinyMatrix& TinyMatrix::Sigmoid() {
    return this->mapInline([](float val, int r, int c) { return 1.0f / (1.0f + std::exp(-val)); }, true);
}

TinyMatrix& TinyMatrix::D_Sigmoid() {
    return this->mapInline([](float val, int r, int c) { return val * (1.0f - val); }, true);
}

TinyMatrix& TinyMatrix::Tanh() {
    return this->mapInline([](float val, int r, int c) { return std::tanh(val); }, true);
}

TinyMatrix& TinyMatrix::D_Tanh() {
    return this->mapInline([](float val, int r, int c) { return 1.0f - (val * val); }, true);
}

TinyMatrix& TinyMatrix::fixed_tanh() {
    return this->mapInline([](float val, int r, int c) {
        int16_t x = (int16_t)val;
        bool is_negative = (x < 0);
        if(is_negative) x = -x;

        int16_t result = (x >= 1024) ? 256 : TinyMatrix::tanh_lut[x];
        if(is_negative) result = -result;

        return (float)result;
        });
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a, TinyMatrix& b) {
    // 1. Snapshot logic to prevent memory aliasing (Protects A.dot(A, B) scenarios)
    _snapshotToRegister(_scratch, &a, _scratchCapacity);
    TinyMatrix* left = _scratch;
    TinyMatrix* right = &b;

    // If the right operand is THIS matrix, snapshot it before we overwrite it
    if(right == this) {
        _snapshotToRegister(_operandScratch, right, _operandScratchCapacity);
        right = _operandScratch;
    }

    int outRows = left->Rows();
    int outCols = right->Cols();
    int innerDim = left->Cols();

    this->Shape(outRows, outCols);

    // 2. Strict Type Promotion Logic restored
    bool leftFloat = left->IsFloat();
    bool rightFloat = right->IsFloat();
    this->isFloat = leftFloat || rightFloat;

    size_t neededAccum = (size_t)(outRows * outCols);
    if(_accumScratch == nullptr || neededAccum > _accumCapacity) {
        delete[] _accumScratch;
        _accumScratch = new float[neededAccum];
        _accumCapacity = neededAccum;
    }

    // Fast memory zeroing instead of allocation
    std::memset(_accumScratch, 0, neededAccum * sizeof(float));
    float* accum = _accumScratch;

    unsigned char* left_data = left->data;
    unsigned char* right_data = right->data;
    int left_cols = left->Cols();
    int right_cols = right->Cols();

    for(int i = 0; i < outRows; i++) {
        for(int k = 0; k < innerDim; k++) {

            // Read and convert the left value EXACTLY ONCE per k-loop
            int left_pos = (i * left_cols + k) * 2;
            float left_val = leftFloat ? halfToFloat(*(uint16_t*)(left_data + left_pos))
                : (float)*(int16_t*)(left_data + left_pos);

            for(int j = 0; j < outCols; j++) {

                // Read continuous right-side memory
                int right_pos = (k * right_cols + j) * 2;
                float right_val = rightFloat ? halfToFloat(*(uint16_t*)(right_data + right_pos))
                    : (float)*(int16_t*)(right_data + right_pos);

                // Accumulate natively at fp32 speed
                accum[i * outCols + j] += left_val * right_val;
            }
        }
    }

    // 4. Strict Memory Packing based on the correct type
    unsigned char* target_data = this->data;
    if(this->isFloat) {
        for(int i = 0; i < outRows * outCols; i++) {
            *(uint16_t*)(target_data + i * 2) = floatToHalf(accum[i]);
        }
    } else {
        for(int i = 0; i < outRows * outCols; i++) {
            *(int16_t*)(target_data + i * 2) = (int16_t)accum[i];
        }
    }

    return *this;
}

// Single argument dot product just proxies to the dual-argument version
TinyMatrix& TinyMatrix::dot(TinyMatrix& a) {
    return this->dot(*this, a);
}

TinyMatrix& TinyMatrix::fixed_dot(TinyMatrix& a, TinyMatrix& b) {
    _snapshotToRegister(_scratch, &a, _scratchCapacity);
    TinyMatrix* left = _scratch;
    TinyMatrix* right = &b;
    if(&b == this) {
        _snapshotToRegister(_operandScratch, &b, _operandScratchCapacity);
        right = _operandScratch;
    }

    int outRows = left->Rows();
    int outCols = right->Cols();
    int innerDim = left->Cols();

    this->Shape(outRows, outCols);
    this->isFloat = false;

    size_t neededAccum = (size_t)(outRows * outCols);
    if(_accumScratchInt == nullptr || neededAccum > _accumIntCapacity) {
        delete[] _accumScratchInt;
        _accumScratchInt = new int32_t[neededAccum];
        _accumIntCapacity = neededAccum;
    }

    // Fast memory zeroing instead of allocation
    std::memset(_accumScratchInt, 0, neededAccum * sizeof(*_accumScratchInt));
    int32_t* accum = _accumScratchInt;

    unsigned char* left_data = left->data;
    unsigned char* right_data = right->data;
    int left_cols = left->Cols();
    int right_cols = right->Cols();

    for(int i = 0; i < outRows; i++) {
        for(int k = 0; k < innerDim; k++) {
            // Read left ONCE per k-loop directly from memory
            int32_t left_val = (int16_t) * (int16_t*)(left_data + (i * left_cols + k) * 2);

            for(int j = 0; j < outCols; j++) {
                // Read right directly from memory
                int32_t right_val = (int16_t) * (int16_t*)(right_data + (k * right_cols + j) * 2);

                accum[i * outCols + j] += (left_val * right_val) >> 8;
            }
        }
    }

    // Write back
    unsigned char* target_data = this->data;
    for(int i = 0; i < outRows * outCols; i++) {
        *(int16_t*)(target_data + i * 2) = (int16_t)accum[i];
    }

    return *this;
}

TinyMatrix& TinyMatrix::QuantizeQ88() {
    this->multiply(256.0f); // Shift float left 8 bits mathematically
    this->Ints();           // Strip float state and truncate to int16_t
    return *this;
}

TinyMatrix& TinyMatrix::DequantizeQ88() {
    // Multiplying an Int matrix by a float automatically promotes it back to fp16!
    this->multiply(1.0f / 256.0f);
    return *this;
}

void TinyMatrix::print(std::string extra) {
    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            if(this->isFloat) {
                printf("%.3f%s", (*this)(i, j), (j == this->Cols() - 1 ? "\n" : " "));
            } else {
                printf("%d%s", (int16_t)(*this)(i, j), (j == this->Cols() - 1 ? "\n" : " "));
            }
        }
    }
    printf("%s", extra.c_str());
}

TinyMatrix& TinyMatrix::transpose() {
    // Snapshot into the dedicated transpose register instead of _scratch!
    _snapshotToRegister(_transposeScratch, this, _transposeScratchCapacity);

    // Flip rows and columns
    this->Shape(_transposeScratch->Cols(), _transposeScratch->Rows());

    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            // Read from (j, i) and write to (i, j)
            (*this)(i, j, (*_transposeScratch)(j, i));
        }
    }
    return *this;
}


nRet TinyMatrix::sum() {
    float total = 0.0f;
    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            total += (*this)(i, j);
        }
    }
    return nRet(total);
}
TinyMatrix& TinyMatrix::Floats() {
    this->isFloat = true;
    return *this;
}
TinyMatrix& TinyMatrix::Randomize(float min, float max) {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            // Generate a random float between min and max
            float r = min + ((float)rand() / (RAND_MAX)) * (max - min);

            if(this->isFloat) {
                (*this)(i, j, r);
            } else {
                (*this)(i, j, (int16_t)r);
            }
        }
    }
    return *this;
}

unsigned char* TinyMatrix::operator[](const int p) {
    //return &this->data[(p)];
    return &this->data[(p * 2)];
}

// =====================================================================
// Capacity-aware scratch register helpers (allocation savings)
// =====================================================================

void TinyMatrix::_ensureScratchCapacity(TinyMatrix*& reg, size_t& capacity, int rows, int cols) {
    size_t needed = size_t(makeEven(rows * cols)) * 2;

    if(reg == nullptr || needed > capacity) {
        delete reg;
        reg = new TinyMatrix(rows, cols);
        capacity = reg->size;
    } else {
        // Buffer is big enough — just update logical dimensions, no allocation
        reg->rows = rows;
        reg->cols = cols;
        reg->size = (int)needed;
    }
}

void TinyMatrix::_snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap) {
    if(!src) return;

    _ensureScratchCapacity(dest, cap, src->rows, src->cols);

    if(dest && src && dest->data && src->data && dest->size >= src->size) {
        std::memcpy(dest->data, src->data, src->size);
        dest->isFloat = src->isFloat;
    }
}

#endif