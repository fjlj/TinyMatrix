#include "TinyMatrix.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <limits>

// Helper to test half-float precision AND ensure the matrix is in Float mode
void assertFloatVal(TinyMatrix& m, int r, int c, float expected, float epsilon = 0.05f) {
    if(!m.IsFloat()) {
        std::cerr << "Type Assertion Failed! Expected Float matrix, but got Int matrix." << std::endl;
        assert(false);
    }

    float actual = m(r, c);
    if(std::abs(actual - expected) > epsilon) {
        std::cerr << "Float Math Failed! Expected: " << expected << " but got: " << actual << std::endl;
        assert(false);
    }
}

// Helper to test integer math AND ensure the matrix is in Int mode
void assertIntVal(TinyMatrix& m, int r, int c, int expected) {
    if(m.IsFloat()) {
        std::cerr << "Type Assertion Failed! Expected Int matrix, but got Float matrix." << std::endl;
        assert(false);
    }

    int actual = (int)m(r, c); // Cast the return value back to int for strict comparison
    if(actual != expected) {
        std::cerr << "Int Math Failed! Expected: " << expected << " but got: " << actual << std::endl;
        assert(false);
    }
}

void TestInitialization() {
    std::cout << "Running TestInitialization..." << std::endl;

    // Test Int Initialization
    TinyMatrix m1(2, 2, {1, 2, 3, 4});
    assert(m1.Rows() == 2 && m1.Cols() == 2);
    assertIntVal(m1, 0, 0, 1);
    assertIntVal(m1, 1, 1, 4);

    // Test Float Initialization
    TinyMatrix m2(2, 2, {1.5, -2.5, 3.14, 4.0});
    assert(m2.Rows() == 2 && m2.Cols() == 2);
    assertFloatVal(m2, 0, 0, 1.5f);
    assertFloatVal(m2, 0, 1, -2.5f);

    // Test Read/Write element access (Int)
    m1(0, 1, 99);
    assertIntVal(m1, 0, 1, 99);

    std::cout << " -> PASS" << std::endl;
}

void TestScalarMath() {
    std::cout << "Running TestScalarMath (Strict Types)..." << std::endl;

    TinyMatrix m_int(2, 2, {10, 20, 30, 40});
    TinyMatrix m_float(2, 2, {1.5, 2.5, 3.5, 4.5});

    // --- Int Matrix Math (Must remain Int) ---
    m_int.add(5);
    assertIntVal(m_int, 0, 0, 15);

    m_int.sub(10);
    assertIntVal(m_int, 0, 0, 5);

    m_int.multiply(2);
    assertIntVal(m_int, 0, 0, 10);

    // --- Float Matrix Math (Must remain Float) ---
    m_float.add(1.5f);
    assertFloatVal(m_float, 0, 0, 3.0f);

    m_float.multiply(2.0f);
    assertFloatVal(m_float, 0, 0, 6.0f);

    std::cout << " -> PASS" << std::endl;
}

void TestMatrixMath() {
    std::cout << "Running TestMatrixMath (A + B, A - B)..." << std::endl;

    // Int Matrices
    TinyMatrix A_int(2, 2, {10, 20, 30, 40});
    TinyMatrix B_int(2, 2, {1,  2,  3,  4});

    // Float Matrices
    TinyMatrix C_flt(2, 2, {1.5, 2.5, 3.5, 4.5});
    TinyMatrix D_flt(2, 2, {0.5, 0.5, 0.5, 0.5});

    // Test Int Addition/Subtraction
    A_int.add(B_int);
    assertIntVal(A_int, 0, 0, 11);

    A_int.sub(B_int);
    assertIntVal(A_int, 0, 0, 10); // Should return to original int value

    // Test Float Addition/Subtraction
    C_flt.add(D_flt);
    assertFloatVal(C_flt, 0, 0, 2.0f);

    std::cout << " -> PASS" << std::endl;
}

void TestDotProduct() {
    std::cout << "Running TestDotProduct (Pointer Aliasing & Types)..." << std::endl;

    TinyMatrix A_int(2, 3, {1, 2, 3, 4, 5, 6});
    TinyMatrix B_int(3, 2, {7, 8, 9, 10, 11, 12});
    TinyMatrix Output_int(1, 1);

    // Basic Int Dot Product
    Output_int.dot(A_int, B_int);
    assertIntVal(Output_int, 0, 0, 58);  // 1*7 + 2*9 + 3*11 = 58
    assertIntVal(Output_int, 1, 1, 154); // 4*8 + 5*10 + 6*12 = 154

    // Test Pointer Aliasing Bug Fix! 
    A_int.dot(A_int, B_int);
    assertIntVal(A_int, 0, 0, 58);

    // Float Dot Product
    TinyMatrix C_flt(2, 2, {1.5, 2.0, 3.5, 4.0});
    TinyMatrix D_flt(2, 2, {2.0, 1.0, 0.5, 2.5});
    C_flt.dot(C_flt, D_flt);
    assertFloatVal(C_flt, 0, 0, 4.0f); // 1.5*2 + 2.0*0.5 = 4.0

    std::cout << " -> PASS" << std::endl;
}

void TestTypePromotion() {
    std::cout << "Running TestTypePromotion..." << std::endl;

    TinyMatrix IntMat(2, 2, {1, 2, 3, 4});
    TinyMatrix FloatMat(2, 2, {0.5, 0.5, 0.5, 0.5});

    // Int matrix + Float matrix = Should promote to Float matrix
    IntMat.add(FloatMat);
    assertFloatVal(IntMat, 0, 0, 1.5f);

    // Test demotion back to Ints
    IntMat.Ints();
    assertIntVal(IntMat, 0, 0, 1); // Truncates 1.5 -> 1 and resets state to Int

    // Float multiplication on an Int matrix should promote to Float
    TinyMatrix IntMat2(2, 2, {10, 20, 30, 40});
    IntMat2.multiply(0.5f);
    assertFloatVal(IntMat2, 0, 0, 5.0f);

    std::cout << " -> PASS" << std::endl;
}

void TestUnusualUses() {
    std::cout << "Running TestUnusualUses (Cryptography, Graph Theory, Data Flattening)..." << std::endl;

    // ---------------------------------------------------------
    // Unusual Use 1: Parallel Cryptography (Caesar Cipher)
    // Matrices don't just hold math; they hold ASCII. By putting 
    // characters into an Int matrix, we can use the scalar math 
    // engine to encrypt/decrypt an entire string simultaneously!
    // ---------------------------------------------------------
    TinyMatrix CipherText(1, 4, {'W', 'O', 'R', 'D'});

    // Encrypt by shifting the ASCII values up by 5
    CipherText.add(5);
    assertIntVal(CipherText, 0, 0, 'W' + 5); // 'W' becomes '\'
    assertIntVal(CipherText, 0, 1, 'O' + 5); // 'O' becomes 'T'

    // Decrypt it back
    CipherText.sub(5);
    assertIntVal(CipherText, 0, 0, 'W');
    assertIntVal(CipherText, 0, 3, 'D');


    // ---------------------------------------------------------
    // Unusual Use 2: Graph Theory (Network Routing)
    // An adjacency matrix represents connections. Node 0 connects 
    // to Node 1, Node 1 to Node 2. Squaring this matrix (Dot Product 
    // with itself) magically calculates the number of 2-step paths!
    // ---------------------------------------------------------
    TinyMatrix Network(3, 3, {
        0, 1, 0,  // Node 0 connects to Node 1
        0, 0, 1,  // Node 1 connects to Node 2
        0, 0, 0   // Node 2 connects to nowhere
        });

    // Dot product of the network with itself
    Network.dot(Network, Network);

    // The engine should automatically calculate that there is exactly 
    // ONE path of length 2 that goes from Node 0 to Node 2 (0 -> 1 -> 2).
    assertIntVal(Network, 0, 2, 1);
    // And zero paths of length 2 from Node 1 to anywhere.
    assertIntVal(Network, 1, 2, 0);


    // ---------------------------------------------------------
    // Unusual Use 3: Data Flattening (1D Stream Conversion)
    // Because your Shape() function maps data sequentially, we can 
    // take a 2D image/grid and instantly flatten it into a 1D 
    // network packet stream without doing any math at all.
    // ---------------------------------------------------------
    TinyMatrix Grid2D(2, 2, {
        11, 22,
        33, 44
        });

    // Flatten to 1 Row, 4 Columns
    Grid2D.Shape(1, 4);

    // Verify the data survived the dimensional shift intact
    assert(Grid2D.Rows() == 1 && Grid2D.Cols() == 4);
    assertIntVal(Grid2D, 0, 0, 11);
    assertIntVal(Grid2D, 0, 2, 33);
    assertIntVal(Grid2D, 0, 3, 44);

    std::cout << " -> PASS" << std::endl;
}
void TestEdgeCases() {
    std::cout << "Running TestEdgeCases (NaN, Infinity, and Deep Copies)..." << std::endl;

    // ---------------------------------------------------------
    // Edge Case 1: Extreme IEEE-754 Float Math
    // The maximum possible value for a 16-bit float is ~65504.0.
    // We will test the max bounds, Positive Infinity, Negative 
    // Infinity, and a pure NaN (Not-a-Number).
    // ---------------------------------------------------------
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    float max_half = 65504.0f;

    TinyMatrix ExtremeMat(2, 2, {inf, -inf, nan, max_half});

    // The floatToHalf and halfToFloat bit-shifters must perfectly 
    // maintain the integrity of these special CPU flags.

    // Check Pos/Neg Infinity
    assert(std::isinf(ExtremeMat(0, 0)));
    assert(ExtremeMat(0, 0) > 0);

    assert(std::isinf(ExtremeMat(0, 1)));
    assert(ExtremeMat(0, 1) < 0);

    // Check pure NaN (std::isnan is required because NaN != NaN)
    assert(std::isnan(ExtremeMat(1, 0)));

    // Check Max bounds
    assertFloatVal(ExtremeMat, 1, 1, 65504.0f);


    // ---------------------------------------------------------
    // Edge Case 2: Memory Wipe via Copy Assignment Operator
    // We will force a float matrix holding Infinity/NaN to 
    // absorb an integer matrix. It must perfectly strip its 
    // float state, avoid heap fragmentation, and become an Int.
    // ---------------------------------------------------------
    TinyMatrix IntMat(2, 2, {1, 2, 3, 4});

    // Trigger operator=
    ExtremeMat = IntMat;

    // Verify the float state was utterly destroyed and replaced
    assert(ExtremeMat.IsFloat() == false);
    assertIntVal(ExtremeMat, 0, 0, 1);
    assertIntVal(ExtremeMat, 1, 1, 4);


    // ---------------------------------------------------------
    // Edge Case 3: Deep Copy Constructor Survival
    // We will copy construct a float matrix and ensure the exact 
    // 16-bit binary state survives the memory clone.
    // ---------------------------------------------------------
    TinyMatrix FloatSource(2, 2, {0.333, 0.666, inf, nan});
    TinyMatrix DeepClone = FloatSource; // Triggers Copy Constructor

    assert(DeepClone.IsFloat() == true);
    assertFloatVal(DeepClone, 0, 0, 0.333f);
    assert(std::isinf(DeepClone(1, 0)));
    assert(std::isnan(DeepClone(1, 1)));

    std::cout << " -> PASS" << std::endl;
}

void TestNeuralNetworkOps() {
    std::cout << "Running TestNeuralNetworkOps (Activations, Derivatives, Hadamard)..." << std::endl;

    // ---------------------------------------------------------
    // 1. ReLU Activation
    // Ensures negatives are clamped to 0, and positives survive.
    // ---------------------------------------------------------
    TinyMatrix M_Relu(2, 2, {-5.0, -0.1, 0.0, 3.5});
    M_Relu.Relu();
    assertFloatVal(M_Relu, 0, 0, 0.0f);
    assertFloatVal(M_Relu, 0, 1, 0.0f);
    assertFloatVal(M_Relu, 1, 0, 0.0f);
    assertFloatVal(M_Relu, 1, 1, 3.5f);

    // ---------------------------------------------------------
    // 2. Sigmoid Activation
    // Ensures the curve correctly squashes values between 0 and 1.
    // ---------------------------------------------------------
    TinyMatrix M_Sig(1, 3, {0.0, 2.0, -2.0});
    M_Sig.Sigmoid();
    assertFloatVal(M_Sig, 0, 0, 0.5f);          // Sigmoid(0) = 0.5
    assertFloatVal(M_Sig, 0, 1, 0.8807f, 0.01f); // Sigmoid(2) ~ 0.88
    assertFloatVal(M_Sig, 0, 2, 0.1192f, 0.01f); // Sigmoid(-2) ~ 0.11

    // ---------------------------------------------------------
    // 3. ReLU Derivative
    // Ensures the gradient is exactly 1 for positives, 0 for negatives.
    // ---------------------------------------------------------
    TinyMatrix M_dRelu(2, 2, {-5.0, 0.0, 0.1, 100.0});
    M_dRelu.D_Relu();
    assertFloatVal(M_dRelu, 0, 0, 0.0f);
    assertFloatVal(M_dRelu, 0, 1, 0.0f); // 0 is generally routed to 0 in dReLU
    assertFloatVal(M_dRelu, 1, 0, 1.0f);
    assertFloatVal(M_dRelu, 1, 1, 1.0f);

    // ---------------------------------------------------------
    // 4. Sigmoid Derivative
    // Crucial Backprop test: Assumes input is an ALREADY ACTIVATED output.
    // Formula: output * (1 - output)
    // ---------------------------------------------------------
    TinyMatrix M_dSig(1, 2, {0.5, 0.8807});
    M_dSig.D_Sigmoid();
    assertFloatVal(M_dSig, 0, 0, 0.25f);          // 0.5 * (1 - 0.5) = 0.25
    assertFloatVal(M_dSig, 0, 1, 0.1049f, 0.01f); // 0.8807 * (1 - 0.8807) ~ 0.105

    // ---------------------------------------------------------
    // 5. Hadamard Product (Element-wise Multiplication)
    // Tests that A * B aligns element-by-element, AND tests that
    // it correctly auto-promotes an Int matrix to a Float matrix.
    // ---------------------------------------------------------
    TinyMatrix H1(2, 2, {1, 2, 3, 4}); // Started as an Int matrix
    TinyMatrix H2(2, 2, {0.5, 1.5, -2.0, 10.0});

    H1.hadamard(H2);

    assert(H1.IsFloat() == true); // Ensure type promotion triggered
    assertFloatVal(H1, 0, 0, 0.5f);  // 1 * 0.5
    assertFloatVal(H1, 0, 1, 3.0f);  // 2 * 1.5
    assertFloatVal(H1, 1, 0, -6.0f); // 3 * -2.0
    assertFloatVal(H1, 1, 1, 40.0f); // 4 * 10.0

    std::cout << " -> PASS" << std::endl;
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "      TinyMatrix Test Suite Started       \n";
    std::cout << "==========================================\n\n";

    TestInitialization();
    TestScalarMath();
    TestMatrixMath();
    TestDotProduct();
    TestTypePromotion();
    TestNeuralNetworkOps();
    TestUnusualUses();
    TestEdgeCases();

    std::cout << "\n==========================================\n";
    std::cout << "  ALL TESTS PASSED! TYPES ARE STRICT!     \n";
    std::cout << "==========================================\n";

    return 0;
}