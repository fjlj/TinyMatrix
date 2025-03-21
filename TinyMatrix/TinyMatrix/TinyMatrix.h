#pragma once
#include <assert.h>
#include <iostream>


enum mapFuncs { TPOS, PRINT, ADDS, SUBS, MULS, DOT, SUM, SUB, ADD,ADDR,SUBR,INTS,MULFS};

typedef uint16_t HALF;

struct mHalf {
    union {
        HALF uVal;
        struct {
            HALF Frac : 10;
            HALF Exp : 5;
            HALF Sign : 1;
        }parts{ 0 };
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
        }parts{ 0 };
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
    int indexMode;
    unsigned char* data;
    bool isFloat = false;


public:

    int Rows() { return this->rows; }
    int Cols() { return this->cols; }
    int Size() { return this->size; }
    bool IsFloat() { return this->isFloat; }
    int IndexMode() { return this->indexMode; }


    TinyMatrix();

    TinyMatrix(int r, int c, int im);

    TinyMatrix(int r, int c, int im, double nums,...);
    TinyMatrix(int r, int c, int im, int nums,...);

    void init(int r, int c, int im);

    ~TinyMatrix();

    TinyMatrix(const TinyMatrix& source);

    TinyMatrix& operator=(const TinyMatrix& source);


    float operator()(int r, int c);
    TinyMatrix& Shape(int r, int c, bool absolute=false);

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
    void print(std::string extra = "");

    TinyMatrix& transpose();
    TinyMatrix& dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& dot(TinyMatrix& a);
    TinyMatrix& Ints();
    nRet sum();

    

private:
    unsigned char* operator[](const int p);
    void map(int n, void* o2_ret = nullptr, TinyMatrix* other = nullptr);
    static HALF floatToHalf(mFloat i);
    static float halfToFloat(mHalf y);

};