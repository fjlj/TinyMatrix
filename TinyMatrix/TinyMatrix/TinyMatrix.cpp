#include "TinyMatrix.h"

constexpr auto makeEven(int a) { return (a + 1 ^ (~a & 1)); }
constexpr auto NOT_VOID = 1;
constexpr auto NEEDS_COPY = 2;
constexpr auto NEEDS_VAL = 4;
constexpr auto NEEDS_OTHER = 8;
constexpr auto RESHAPE = 16;
constexpr auto AS_FLOAT = 32;

float print(float val, int r, int c, TinyMatrix* _this) {
    if (_this->IsFloat()) {
        printf("%.3f%s", val, ((c == 0 && r == 0) ? "\n" : (c == 0 ? "\n" : " ")));
    }
    else {
        printf("%d%s", (int16_t)val, ((c == 0 && r == 0) ? "\n" : (c == 0 ? "\n" : " ")));
    }
    return 0.0f;
};


float tpos(float val, float r, int c, TinyMatrix* d) {
    //int pos = (r * (d->Cols()) + c);
    return (*d)(c, (int)r);
}

float add(float val, float r, int c, TinyMatrix* _notused) {
    return val + r;
};

float add2(float val, float r, int c, TinyMatrix* d) {
    return val + (float)(*d)((int)r, c);
};

float sub(float val, float s, int _notused, TinyMatrix* _notused2) {
    return val - s;
};

float sub2(float val, float r, int c, TinyMatrix* d) {
    return (float)(*d)((int)r, c) - val;
};

float mul(float val, float s, int _notused, TinyMatrix* _notused2) {
    return val * s;
};

void* mFuncs[13] = { 
    &tpos,  //TPOS
    &print, //PRINT 
    &add,   //ADDS
    &sub,   //SUBS
    &mul,   //MULS
    &mul,   //DOT
    &mul,   //SUM
    &sub2,  //SUB
    &add2,  //ADD
    &add2,  //ADDR
    &sub2,  //SUBR
    &add2,  //INTS
    &mul    //MULFS
};
char  mFuncs_t[13] = {
    (NOT_VOID | NEEDS_COPY),                        //Transpose
    (NEEDS_VAL),                                    //Print
    (NOT_VOID | NEEDS_VAL),                         //Add scalar ADDS
    (NOT_VOID | NEEDS_VAL),                         //Subtract scalar
    (NOT_VOID | NEEDS_VAL),                         //Multiply scalar
    (NOT_VOID | NEEDS_OTHER),                       //Dot Product of 2 matricies or 1 matrix (autotransposes 2nd matrix or self if only 1 provided)
    (NEEDS_VAL),                                    //Summation of matrix
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           //Subtract 1 matrix from self
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           //add matrix to self
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER | RESHAPE), //Subtract 1 matrix from self (reshape fill empty with 0s)
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER | RESHAPE), //add matrix to self (reshape fill empty with 0s)
    (NOT_VOID | NEEDS_COPY),                         //converts half float matrix back to int16_t matrix
    (NOT_VOID | NEEDS_VAL | AS_FLOAT)                //multiply by float scalar...
};



TinyMatrix::TinyMatrix() {
    this->rows = 1; this->cols = 1; this->size = 4;
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::TinyMatrix(int r, int c) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<double> nums) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
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
    this->data = new unsigned char[this->size] { 0 };
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
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::~TinyMatrix() {
    delete[] this->data;
}

TinyMatrix::TinyMatrix(const TinyMatrix& source) {
    this->init(source.rows, source.cols);
    this->isFloat = source.isFloat;
    std::copy(source.data, source.data + source.size, this->data);
}

TinyMatrix& TinyMatrix::operator=(const TinyMatrix& source) {
    if(&source == this) return *this;
    this->init(source.rows, source.cols);
    this->isFloat = source.isFloat;
    std::copy(source.data, (source.data + source.size), this->data);
    return *this;
}

float TinyMatrix::operator()(int r, int c) {
    int pos = (r * (this->cols) + c);
    return (this->isFloat ? halfToFloat((uint16_t) * (uint16_t*)((*this)[pos])) : (float)*(int16_t*)((*this)[pos]));
}

TinyMatrix& TinyMatrix::Shape(int r, int c, bool absolute) {
    if(this->rows == r && this->cols == c) return *this;

    int old_r = this->rows; int old_c = this->cols; int old_size = this->size;
    this->rows = r; this->cols = c;
    unsigned char* old_data = &this->data[0];
    this->size = (makeEven((r * c)) * 2);

    bool olim = (old_size <= this->size);
    if(absolute) {
        this->data = new unsigned char[this->size] { 0 };
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
        delete[] old_data; // FIXED!
    } else {
        this->data = new unsigned char[this->size] { 0 };
        std::copy(old_data, old_data + (old_size < this->size ? old_size : this->size), this->data);
        delete[] old_data; // FIXED!
    }
    return *this;
}

void TinyMatrix::operator()(int r, int c, const int v) {
    if(this->isFloat) {
        (*this)(r, c, (float)v); return;
    }
    int pos = (r * (this->cols) + c);
    *(int16_t*)(*this)[pos] = (int16_t)v;
}

void TinyMatrix::operator()(int r, int c, const double v) {
    TinyMatrix::operator()(r, c,(float)v);
}

void TinyMatrix::operator()(int r, int c, const float v) {
    this->isFloat = true;
    int pos = (r * (this->cols) + c);
    *(uint16_t*)(*this)[pos] = floatToHalf(v);
}

float TinyMatrix::halfToFloat(mHalf y)
{

    mFloat ret(0.0f);
    ret.parts.Sign = y.parts.Sign;

    if (!y.parts.Exp) {
        if (!y.parts.Frac) {
            ret.parts.Exp = 0;
            ret.parts.Frac = 0;
        }
        else {
            const float half_denorm = (1.0f / 16384.0f);
            float mantissa = ((float)y.parts.Frac) / 1024.0f;
            float sgn = (((float)y.parts.Sign) * 2.0f) - 1.0f;
            ret.fVal = sgn * mantissa * half_denorm;
        }
    }
    else if (y.parts.Exp == 31) {
        ret.parts.Exp = 0xff;
        ret.parts.Frac = (y.parts.Frac != 0);
    }
    else {

        ret.parts.Exp = (uint32_t)y.parts.Exp + (127 - 15);
        ret.parts.Frac = (uint32_t)y.parts.Frac << 13;
    }

    return ret.fVal;
}

HALF TinyMatrix::floatToHalf(mFloat i)
{
    mHalf ret(0);
    ret.parts.Sign = i.parts.Sign;
    int _3precision = (int)floor((i.fVal * 1000.0f));
    float nf = _3precision / 999.875f;
    i.fVal = (nf + 0.00006103515625f);

    register int e = i.parts.Exp - 112;
    register int f = i.parts.Frac;

    if (e < 0) {
        if (e < -10)
            return ret.uVal;

        f = (f | 0x00800000) >> (1 - e);
        ret.parts.Frac = (f >> 13);
        return ret.uVal;

    }
    else if (e == 0xff - (112)) {
        ret.parts.Frac = (i.parts.Frac != 0);
        ret.parts.Exp = 31;
        return ret.uVal;

    }
    else {
        if (e > 30) {
            ret.parts.Exp = 31;
            return ret.uVal;
        }

        ret.parts.Exp = e;
        ret.parts.Frac =(f >> 13);
        return ret.uVal;
    }
}

TinyMatrix& TinyMatrix::Ints()
{
    if (!this->isFloat)
        return *this;

    this->map(mapFuncs::INTS);
    return *this;
}

TinyMatrix& TinyMatrix::add(int s)
{
    float as = (float)s;
    this->map(mapFuncs::ADDS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::sub(int s)
{
    float as = (float)s;
    this->map(mapFuncs::SUBS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::add(float s)
{
    float as = s;
    this->map(mapFuncs::ADDS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::sub(float s)
{
    float as = s;
    this->map(mapFuncs::SUBS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::add(double s)
{
    this->add((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::sub(double s)
{
    this->sub((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::add(const TinyMatrix& a, bool reshape)
{
    if (reshape) {
        this->map(mapFuncs::ADDR, (void*)&a);
    }
    else {
        this->map(mapFuncs::ADD, (void*)&a);
    }
    return *this;
}

TinyMatrix& TinyMatrix::sub(const TinyMatrix& a, bool reshape)
{
    if (reshape) {
        this->map(mapFuncs::SUBR, (void*)&a);
    }
    else {
        this->map(mapFuncs::SUB, (void*)&a);
    }
    return *this;
}

TinyMatrix& TinyMatrix::multiply(int s)
{
    float as = (float)s;
    this->map(mapFuncs::MULS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::multiply(float s)
{
    float as = (float)s;
    if (this->isFloat) {
        this->map(mapFuncs::MULS, &as);
    }
    else {
        this->map(mapFuncs::MULFS, &as);
    }
    return *this;
}

TinyMatrix& TinyMatrix::multiply(double s)
{
    this->multiply((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a, TinyMatrix& b)
{
  
    this->map(mapFuncs::DOT, &a, &b);
    return *this;
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a)
{

    this->map(mapFuncs::DOT, &a, nullptr);
    return *this;
}

void TinyMatrix::print(std::string extra)
{
    this->map(mapFuncs::PRINT);
    printf("%s", extra.c_str());
}


TinyMatrix& TinyMatrix::transpose()
{
    this->map(mapFuncs::TPOS);
    return *this;
}

nRet TinyMatrix::sum()
{
    nRet as;
    this->map(mapFuncs::SUM, &as);
    return as;
}

void TinyMatrix::map(int n, void* o2_ret, TinyMatrix* other) {
    TinyMatrix* scratchNib = nullptr;
    TinyMatrix* o2_copy = nullptr;
    float val = 0;
    float sum = 0;

    float(*foo)(float, float, int, TinyMatrix*);
    foo = (float(*)(float, float, int, TinyMatrix*))mFuncs[n];


    if (mFuncs_t[n] & (NEEDS_COPY | NEEDS_OTHER)) {
        if (mFuncs_t[n] & NEEDS_COPY)
            scratchNib = new TinyMatrix(*this);
        if (mFuncs_t[n] & NEEDS_OTHER && !(mFuncs_t[n] & NEEDS_COPY)) {
            assert(other != nullptr || o2_ret != nullptr);
            if (o2_ret != nullptr && other == nullptr)
                other = (TinyMatrix*)o2_ret;
            scratchNib = new TinyMatrix(*other);
            if (n == mapFuncs::DOT) {
                if (scratchNib->rows != ((TinyMatrix*)o2_ret)->cols && scratchNib->cols == ((TinyMatrix*)o2_ret)->cols) {
                    //other matrix was provided but not transposed or DOT with self
                    scratchNib->map(mapFuncs::TPOS);
                }
            }
        }
        if (n == mapFuncs::TPOS) {
            this->Shape(this->cols, this->rows);
        }
        if (n == mapFuncs::INTS) {
            this->isFloat = false;
        }
        assert(scratchNib != nullptr);
    }
    TinyMatrix* leftOperand = nullptr;
    bool deleteLeftOperand = false;

    if(n == mapFuncs::DOT) {
        assert(o2_ret != nullptr);
        TinyMatrix* o2_matrix = (TinyMatrix*)o2_ret;

        if(this == o2_matrix) {
            leftOperand = new TinyMatrix(*o2_matrix);
            deleteLeftOperand = true;
        } else {
            leftOperand = o2_matrix;
        }

        if(this->cols != scratchNib->cols || this->rows != leftOperand->rows) {
            this->Shape(leftOperand->rows, scratchNib->cols);
        }
    }
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            if (mFuncs_t[n] & NEEDS_VAL) {
                if (mFuncs_t[n] & AS_FLOAT) this->isFloat = false;
                val = (*this)(i, j);
            }
            if (mFuncs_t[n] & NOT_VOID) {
                if (scratchNib != nullptr) {
                    switch (n) {
                    case mapFuncs::DOT:
                        sum = 0;
                        // EXACT FIX 2: Use leftOperand to read data safely
                        for(int a = 0; a < leftOperand->cols; a++) {
                            sum += foo((*scratchNib)(a, j), (*leftOperand)(i, a), 0, nullptr);
                        }

                        if(this->isFloat) {
                            (*this)(i, j, sum);
                        } else {
                            (*this)(i, j, (int16_t)sum);
                        }
                        break;
                    case mapFuncs::SUB:
                    case mapFuncs::ADD:
                    case mapFuncs::SUBR:
                    case mapFuncs::ADDR:
                        assert(o2_ret != nullptr);
                        if (o2_copy == nullptr && (mFuncs_t[n] & RESHAPE) || ((TinyMatrix*)o2_ret)->rows != scratchNib->rows && ((TinyMatrix*)o2_ret)->cols != scratchNib->cols) {
                            o2_copy = new TinyMatrix(*(TinyMatrix*)o2_ret);
                            o2_copy->Shape(scratchNib->rows, scratchNib->cols, true);
                        }
                        else if(o2_copy == nullptr) {
                            o2_copy = (TinyMatrix*)o2_ret;
                        }
                        assert(o2_copy->rows == scratchNib->rows && o2_copy->cols == scratchNib->cols);
                    default:
                        TinyMatrix* fromMatrix = (o2_copy != nullptr ? o2_copy : scratchNib);
                        if ((mFuncs_t[n] & NEEDS_COPY) && (mFuncs_t[n] & NEEDS_OTHER))
                            val = (*fromMatrix)(i, j);
                        if (this->isFloat || fromMatrix->isFloat && n != mapFuncs::INTS) {
                            (*this)(i, j, foo(val, (float)i, j, scratchNib));
                        }
                        else {
                            (*this)(i, j, (int16_t)foo(val, (float)i, j, scratchNib));
                        }
                        break;
                    }
                } 
                else {
                    switch (n) {
                    case mapFuncs::ADDS:
                    case mapFuncs::SUBS:
                    case mapFuncs::MULS:
                    case mapFuncs::MULFS:
                    default:
                        if (this->isFloat || (mFuncs_t[n] & AS_FLOAT)) {
                            (*this)(i, j, foo(val, *(float*)o2_ret, 0, nullptr));
                        }
                        else {
                            (*this)(i, j, (int16_t)foo(val, (int16_t)(*(float*)o2_ret), 0, nullptr));
                        }
                        break;
                    }

                }
            } 
            else {
                if (n == mapFuncs::PRINT)
                    foo(val, (float)(this->rows - i - 1), (this->cols - j - 1), this);
            
                if (n == mapFuncs::SUM) 
                    *(nRet*)o2_ret += val;
            }
        }
    }

    if (scratchNib != nullptr)
        delete scratchNib;
    if (o2_copy != nullptr && o2_copy != o2_ret)
        delete o2_copy;
    if (deleteLeftOperand)
        delete leftOperand;
}

unsigned char* TinyMatrix::operator[](const int p) {
    //return &this->data[(p)];
    return &this->data[(p * 2)];
}

