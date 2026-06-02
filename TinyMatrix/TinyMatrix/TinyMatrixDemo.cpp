#define TINYMATRIX_IMPLEMENTATION
#include "TinyMatrix.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <limits>

// ==================== Error Tracking ====================

int g_primary_errors = 0;
int g_cascade_errors = 0;

// ==================== Improved Check Helpers ====================

// Per-test error tracking helper
struct TestContext {
    bool had_error = false;
    std::string test_name;

    explicit TestContext(std::string name): test_name(std::move(name)) {
    }

    bool Check(bool condition, const std::string& message) {
        if(condition) return true;

        if(had_error) {
            std::cerr << "  [CASCADE] " << message << std::endl;
            g_cascade_errors++;
        } else {
            std::cerr << "  [ERROR]   " << message << std::endl;
            g_primary_errors++;
            had_error = true;
        }
        return false;
    }

    bool CheckFloatVal(TinyMatrix& m, int r, int c, float expected, float epsilon = 0.05f) {
        if(!m.IsFloat()) {
            return Check(false, "Expected Float matrix but got Int at (" +
                std::to_string(r) + "," + std::to_string(c) + ")");
        }
        float actual = m(r, c);
        if(std::abs(actual - expected) > epsilon) {
            return Check(false, "Float mismatch at (" + std::to_string(r) + "," + std::to_string(c) +
                "): expected " + std::to_string(expected) +
                " but got " + std::to_string(actual));
        }
        return true;
    }

    bool CheckIntVal(TinyMatrix& m, int r, int c, int expected) {
        if(m.IsFloat()) {
            return Check(false, "Expected Int matrix but got Float at (" +
                std::to_string(r) + "," + std::to_string(c) + ")");
        }
        int actual = (int)m(r, c);
        if(actual != expected) {
            return Check(false, "Int mismatch at (" + std::to_string(r) + "," + std::to_string(c) +
                "): expected " + std::to_string(expected) +
                " but got " + std::to_string(actual));
        }
        return true;
    }

    void PrintResult() const {
        if(!had_error) {
            std::cout << " -> PASS" << std::endl;
        } else {
            std::cout << " -> FAIL (see errors above)" << std::endl;
        }
    }
};

// ==================== Tests ====================

void TestInitialization() {
    TestContext ctx("TestInitialization");
    std::cout << "Running TestInitialization..." << std::endl;

    TinyMatrix m1(2, 2, {1, 2, 3, 4});
    ctx.Check(m1.Rows() == 2 && m1.Cols() == 2, "Matrix dimensions incorrect");
    ctx.CheckIntVal(m1, 0, 0, 1);
    ctx.CheckIntVal(m1, 1, 1, 4);

    TinyMatrix m2(2, 2, {1.5, -2.5, 3.14, 4.0});
    ctx.Check(m2.Rows() == 2 && m2.Cols() == 2, "Float matrix dimensions incorrect");
    ctx.CheckFloatVal(m2, 0, 0, 1.5f);
    ctx.CheckFloatVal(m2, 0, 1, -2.5f);

    m1(0, 1, 99);
    ctx.CheckIntVal(m1, 0, 1, 99);

    ctx.PrintResult();
}

void TestScalarMath() {
    TestContext ctx("TestScalarMath");
    std::cout << "Running TestScalarMath..." << std::endl;

    TinyMatrix m_int(2, 2, {10, 20, 30, 40});
    TinyMatrix m_float(2, 2, {1.5, 2.5, 3.5, 4.5});

    m_int.add(5);
    ctx.CheckIntVal(m_int, 0, 0, 15);

    m_int.sub(10);
    ctx.CheckIntVal(m_int, 0, 0, 5);

    m_int.multiply(2);
    ctx.CheckIntVal(m_int, 0, 0, 10);

    m_float.add(1.5f);
    ctx.CheckFloatVal(m_float, 0, 0, 3.0f);

    m_float.multiply(2.0f);
    ctx.CheckFloatVal(m_float, 0, 0, 6.0f);

    ctx.PrintResult();
}

void TestMatrixMath() {
    TestContext ctx("TestMatrixMath");
    std::cout << "Running TestMatrixMath..." << std::endl;

    TinyMatrix A_int(2, 2, {10, 20, 30, 40});
    TinyMatrix B_int(2, 2, {1, 2, 3, 4});
    TinyMatrix C_flt(2, 2, {1.5, 2.5, 3.5, 4.5});
    TinyMatrix D_flt(2, 2, {0.5, 0.5, 0.5, 0.5});

    A_int.add(B_int);
    ctx.CheckIntVal(A_int, 0, 0, 11);

    A_int.sub(B_int);
    ctx.CheckIntVal(A_int, 0, 0, 10);

    C_flt.add(D_flt);
    ctx.CheckFloatVal(C_flt, 0, 0, 2.0f);

    ctx.PrintResult();
}

void TestDotProduct() {
    TestContext ctx("TestDotProduct");
    std::cout << "Running TestDotProduct..." << std::endl;

    TinyMatrix A_int(2, 3, {1, 2, 3, 4, 5, 6});
    TinyMatrix B_int(3, 2, {7, 8, 9, 10, 11, 12});
    TinyMatrix Output_int(1, 1);

    Output_int.dot(A_int, B_int);
    ctx.CheckIntVal(Output_int, 0, 0, 58);
    ctx.CheckIntVal(Output_int, 1, 1, 154);

    // Aliasing test
    A_int.dot(A_int, B_int);
    ctx.CheckIntVal(A_int, 0, 0, 58);

    TinyMatrix C_flt(2, 2, {1.5, 2.0, 3.5, 4.0});
    TinyMatrix D_flt(2, 2, {2.0, 1.0, 0.5, 2.5});
    C_flt.dot(C_flt, D_flt);
    ctx.CheckFloatVal(C_flt, 0, 0, 4.0f);

    ctx.PrintResult();
}

void TestTypePromotion() {
    TestContext ctx("TestTypePromotion");
    std::cout << "Running TestTypePromotion..." << std::endl;

    TinyMatrix IntMat(2, 2, {1, 2, 3, 4});
    TinyMatrix FloatMat(2, 2, {0.5, 0.5, 0.5, 0.5});

    IntMat.add(FloatMat);
    ctx.CheckFloatVal(IntMat, 0, 0, 1.5f);

    IntMat.Ints();
    ctx.CheckIntVal(IntMat, 0, 0, 1);

    TinyMatrix IntMat2(2, 2, {10, 20, 30, 40});
    IntMat2.multiply(0.5f);
    ctx.CheckFloatVal(IntMat2, 0, 0, 5.0f);

    ctx.PrintResult();
}

void TestNeuralNetworkOps() {
    TestContext ctx("TestNeuralNetworkOps");
    std::cout << "Running TestNeuralNetworkOps..." << std::endl;

    TinyMatrix M_Relu(2, 2, {-5.0, -0.1, 0.0, 3.5});
    M_Relu.Relu();
    ctx.CheckFloatVal(M_Relu, 0, 0, 0.0f);
    ctx.CheckFloatVal(M_Relu, 0, 1, 0.0f);
    ctx.CheckFloatVal(M_Relu, 1, 0, 0.0f);
    ctx.CheckFloatVal(M_Relu, 1, 1, 3.5f);

    TinyMatrix M_Sig(1, 3, {0.0, 2.0, -2.0});
    M_Sig.Sigmoid();
    ctx.CheckFloatVal(M_Sig, 0, 0, 0.5f);
    ctx.CheckFloatVal(M_Sig, 0, 1, 0.8807f, 0.01f);
    ctx.CheckFloatVal(M_Sig, 0, 2, 0.1192f, 0.01f);

    TinyMatrix M_dRelu(2, 2, {-5.0, 0.0, 0.1, 100.0});
    M_dRelu.D_Relu();
    ctx.CheckFloatVal(M_dRelu, 0, 0, 0.0f);
    ctx.CheckFloatVal(M_dRelu, 0, 1, 0.0f);
    ctx.CheckFloatVal(M_dRelu, 1, 0, 1.0f);
    ctx.CheckFloatVal(M_dRelu, 1, 1, 1.0f);

    TinyMatrix M_dSig(1, 2, {0.5, 0.8807});
    M_dSig.D_Sigmoid();
    ctx.CheckFloatVal(M_dSig, 0, 0, 0.25f);
    ctx.CheckFloatVal(M_dSig, 0, 1, 0.1049f, 0.01f);

    TinyMatrix H1(2, 2, {1, 2, 3, 4});
    TinyMatrix H2(2, 2, {0.5, 1.5, -2.0, 10.0});
    H1.hadamard(H2);
    ctx.Check(H1.IsFloat(), "Hadamard should promote Int matrix to Float");
    ctx.CheckFloatVal(H1, 0, 0, 0.5f);
    ctx.CheckFloatVal(H1, 0, 1, 3.0f);
    ctx.CheckFloatVal(H1, 1, 0, -6.0f);
    ctx.CheckFloatVal(H1, 1, 1, 40.0f);

    ctx.PrintResult();
}

void TestUnusualUses() {
    TestContext ctx("TestUnusualUses");
    std::cout << "Running TestUnusualUses..." << std::endl;

    TinyMatrix CipherText(1, 4, {'W', 'O', 'R', 'D'});
    CipherText.add(5);
    ctx.CheckIntVal(CipherText, 0, 0, 'W' + 5);
    ctx.CheckIntVal(CipherText, 0, 1, 'O' + 5);
    CipherText.sub(5);
    ctx.CheckIntVal(CipherText, 0, 0, 'W');
    ctx.CheckIntVal(CipherText, 0, 3, 'D');

    TinyMatrix Network(3, 3, {0, 1, 0, 0, 0, 1, 0, 0, 0});
    Network.dot(Network, Network);
    ctx.CheckIntVal(Network, 0, 2, 1);
    ctx.CheckIntVal(Network, 1, 2, 0);

    TinyMatrix Grid2D(2, 2, {11, 22, 33, 44});
    Grid2D.Shape(1, 4);
    ctx.Check(Grid2D.Rows() == 1 && Grid2D.Cols() == 4, "Shape() failed to flatten matrix");
    ctx.CheckIntVal(Grid2D, 0, 0, 11);
    ctx.CheckIntVal(Grid2D, 0, 2, 33);
    ctx.CheckIntVal(Grid2D, 0, 3, 44);

    ctx.PrintResult();
}

void TestEdgeCases() {
    TestContext ctx("TestEdgeCases");
    std::cout << "Running TestEdgeCases..." << std::endl;

    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    float max_half = 65504.0f;

    TinyMatrix ExtremeMat(2, 2, {inf, -inf, nan, max_half});

    ctx.Check(std::isinf(ExtremeMat(0, 0)) && ExtremeMat(0, 0) > 0, "Positive infinity failed");
    ctx.Check(std::isinf(ExtremeMat(0, 1)) && ExtremeMat(0, 1) < 0, "Negative infinity failed");
    ctx.Check(std::isnan(ExtremeMat(1, 0)), "NaN handling failed");
    ctx.CheckFloatVal(ExtremeMat, 1, 1, max_half);

    TinyMatrix IntMat(2, 2, {1, 2, 3, 4});
    ExtremeMat = IntMat;
    ctx.Check(!ExtremeMat.IsFloat(), "Copy assignment should have demoted to Int");
    ctx.CheckIntVal(ExtremeMat, 0, 0, 1);
    ctx.CheckIntVal(ExtremeMat, 1, 1, 4);

    TinyMatrix FloatSource(2, 2, {0.333, 0.666, inf, nan});
    TinyMatrix DeepClone = FloatSource;
    ctx.Check(DeepClone.IsFloat(), "Copy constructor lost float state");
    ctx.CheckFloatVal(DeepClone, 0, 0, 0.333f);
    ctx.Check(std::isinf(DeepClone(1, 0)), "Infinity lost during deep copy");
    ctx.Check(std::isnan(DeepClone(1, 1)), "NaN lost during deep copy");

    ctx.PrintResult();
}

void TestFixedPointQ88() {
    TestContext ctx("TestFixedPointQ88");
    std::cout << "Running TestFixedPointQ88..." << std::endl;

    TinyMatrix M(2, 2, {0.5, 1.0, -1.5, 2.25});
    M.QuantizeQ88();
    ctx.Check(!M.IsFloat(), "QuantizeQ88 should produce Int matrix");
    ctx.CheckIntVal(M, 0, 0, 128);
    ctx.CheckIntVal(M, 0, 1, 256);
    ctx.CheckIntVal(M, 1, 0, -384);

    TinyMatrix Half(2, 2, {0.5, 0.5, 0.5, 0.5});
    Half.QuantizeQ88();
    M.fixed_hadamard(Half);
    ctx.Check(!M.IsFloat(), "fixed_hadamard should stay in Int mode");
    ctx.CheckIntVal(M, 0, 0, 64);

    M.DequantizeQ88();
    ctx.Check(M.IsFloat(), "DequantizeQ88 should promote back to Float");
    ctx.CheckFloatVal(M, 0, 0, 0.25f);

    ctx.PrintResult();
}

// ==================== NEW TESTS FOR CAPACITY & STRIDE ====================

void TestReserveAndExpand() {
    TestContext ctx("TestReserveAndExpand");
    std::cout << "Running TestReserveAndExpand..." << std::endl;

    TinyMatrix m(2, 2, {1, 2, 3, 4});

    // Reserve extra capacity
    m.Reserve(8, 8);                    // Should allocate bigger buffer
    ctx.Check(m.Rows() == 2 && m.Cols() == 2, "Reserve should not change logical size");

    // Expand logical size into reserved capacity (fast path)
    m.SetLogicalBounds(4, 4);
    ctx.Check(m.Rows() == 4 && m.Cols() == 4, "SetLogicalBounds failed");

    // New area should be zeroed
    ctx.CheckIntVal(m, 2, 2, 0);
    ctx.CheckIntVal(m, 3, 3, 0);

    // Original data should still be there
    ctx.CheckIntVal(m, 0, 0, 1);
    ctx.CheckIntVal(m, 1, 1, 4);

    // Write into newly expanded area
    m(3, 3, 99);
    ctx.CheckIntVal(m, 3, 3, 99);

    ctx.PrintResult();
}

void TestShapeFastExpansion() {
    TestContext ctx("TestShapeFastExpansion");
    std::cout << "Running TestShapeFastExpansion..." << std::endl;

    TinyMatrix m(2, 3);
    m.Reserve(10, 10);

    // This should use the fast in-place expansion path (no allocation)
    m.Shape(5, 5);
    ctx.Check(m.Rows() == 5 && m.Cols() == 5, "Shape fast expansion failed");

    // Should still be able to write everywhere
    m(4, 4, 42.0f);
    ctx.CheckFloatVal(m, 4, 4, 42.0f);

    ctx.PrintResult();
}

void TestShrinkToFit() {
    TestContext ctx("TestShrinkToFit");
    std::cout << "Running TestShrinkToFit..." << std::endl;

    TinyMatrix m(2, 2, {10, 20, 30, 40});
    m.Reserve(10, 10);
    m.SetLogicalBounds(6, 6);

    // Fill some data in the expanded area
    m(5, 5, 777);

    // Now shrink back to logical size
    m.ShrinkToFit();
    ctx.Check(m.Rows() == 6 && m.Cols() == 6, "ShrinkToFit changed logical size unexpectedly");

    // Data should survive compaction
    ctx.CheckIntVal(m, 0, 0, 10);
    ctx.CheckIntVal(m, 5, 5, 777);

    ctx.PrintResult();
}

void TestStridedMatrixOperations() {
    TestContext ctx("TestStridedMatrixOperations");
    std::cout << "Running TestStridedMatrixOperations..." << std::endl;

    TinyMatrix m(2, 2, {1, 2, 3, 4});
    m.Reserve(4, 8);           // stride will be 8 after expansion
    m.SetLogicalBounds(3, 4);

    // Fill the logical 3x4 area
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 4; j++) {
            m(i, j, (i * 10 + j));
        }
    }

    // Test element-wise operations still work correctly with stride
    TinyMatrix addend(3, 4);
    addend.Floats();
    for(int i = 0; i < 3; i++) for(int j = 0; j < 4; j++) addend(i, j, 100.0f);

    m.add(addend);
    ctx.CheckFloatVal(m, 0, 0, 100.0f);
    ctx.CheckFloatVal(m, 2, 3, 123.0f);

    // Test dot product on strided matrix
    TinyMatrix A(2, 3);
    A.Reserve(5, 10);
    A.SetLogicalBounds(2, 3);
    A(0, 0, 1); A(0, 1, 2); A(0, 2, 3);
    A(1, 0, 4); A(1, 1, 5); A(1, 2, 6);

    TinyMatrix B(3, 2, {7,8,9,10,11,12});
    TinyMatrix Out(1, 1);
    Out.dot(A, B);

    ctx.CheckIntVal(Out, 0, 0, 58);
    ctx.CheckIntVal(Out, 1, 1, 154);

    ctx.PrintResult();
}

void TestWriteReadRawWithCapacity() {
    TestContext ctx("TestWriteReadRawWithCapacity");
    std::cout << "Running TestWriteReadRawWithCapacity..." << std::endl;

    TinyMatrix original(2, 2, {1.5f, 2.5f, 3.5f, 4.5f});
    original.Reserve(8, 8);
    original.SetLogicalBounds(4, 4);
    original(3, 3, 9.9f);

    // Write to disk (should call ShrinkToFit internally)
    std::ofstream out("test_matrix.bin", std::ios::binary);
    original.WriteRaw(out);
    out.close();

    // Read back
    TinyMatrix loaded(4, 4);
    std::ifstream in("test_matrix.bin", std::ios::binary);
    loaded.ReadRaw(in);
    in.close();

    ctx.Check(loaded.Rows() == 4 && loaded.Cols() == 4, "ReadRaw size mismatch");
    ctx.CheckFloatVal(loaded, 0, 0, 1.5f);
    ctx.CheckFloatVal(loaded, 3, 3, 9.9f);

    // Cleanup test file
    std::remove("test_matrix.bin");

    ctx.PrintResult();
}

void TestCopyWithStride() {
    TestContext ctx("TestCopyWithStride");
    std::cout << "Running TestCopyWithStride..." << std::endl;

    TinyMatrix src(2, 2, {10, 20, 30, 40});
    src.Reserve(6, 6);
    src.SetLogicalBounds(4, 4);
    src(3, 3, 99);

    // Copy constructor
    TinyMatrix copy = src;
    ctx.Check(copy.Rows() == 4 && copy.Cols() == 4, "Copy constructor size wrong");
    ctx.CheckIntVal(copy, 3, 3, 99);

    // Assignment operator
    TinyMatrix assigned(1, 1);
    assigned = src;
    ctx.Check(assigned.Rows() == 4 && assigned.Cols() == 4, "Assignment size wrong");
    ctx.CheckIntVal(assigned, 0, 0, 10);
    ctx.CheckIntVal(assigned, 3, 3, 99);

    ctx.PrintResult();
}

// ==================== EDGE CASE TESTS ====================

void TestReserveEdgeCases() {
    TestContext ctx("TestReserveEdgeCases");
    std::cout << "Running TestReserveEdgeCases..." << std::endl;

    TinyMatrix m(3, 3, {1,2,3,4,5,6,7,8,9});

    // Edge: Reserve with smaller values (should be no-op)
    m.Reserve(2, 2);
    ctx.Check(m.Rows() == 3 && m.Cols() == 3, "Reserve with smaller size changed logical dimensions");

    // Edge: Reserve with same size (should be no-op)
    m.Reserve(3, 3);
    ctx.Check(m.Rows() == 3 && m.Cols() == 3, "Reserve with equal size changed logical dimensions");

    // Edge: Reserve much larger
    m.Reserve(100, 50);
    ctx.Check(m.Rows() == 3 && m.Cols() == 3, "Large Reserve changed logical size unexpectedly");

    // Now expand into the large reserved space
    m.SetLogicalBounds(20, 20);
    ctx.Check(m.Rows() == 20 && m.Cols() == 20, "Expand after large Reserve failed");

    // Verify original data survived
    ctx.CheckIntVal(m, 0, 0, 1);
    ctx.CheckIntVal(m, 2, 2, 9);

    // New area should be zero-initialized
    ctx.CheckIntVal(m, 10, 10, 0);
    ctx.CheckIntVal(m, 19, 19, 0);

    ctx.PrintResult();
}

void TestSetLogicalBoundsEdgeCases() {
    TestContext ctx("TestSetLogicalBoundsEdgeCases");
    std::cout << "Running TestSetLogicalBoundsEdgeCases..." << std::endl;

    TinyMatrix m(2, 2, {10, 20, 30, 40});

    // Edge: Expand without Reserve (uses internal capacity)
    m.SetLogicalBounds(4, 4);
    ctx.Check(m.Rows() == 4 && m.Cols() == 4, "Expand without prior Reserve failed");
    ctx.CheckIntVal(m, 0, 0, 10);   // Original data
    ctx.CheckIntVal(m, 3, 3, 0);    // New area zeroed

    // Edge: Expand to same size (should be safe)
    m.SetLogicalBounds(4, 4);
    ctx.Check(m.Rows() == 4 && m.Cols() == 4, "Expand to same size broke matrix");

    // Edge: Shrink logical size within existing capacity
    m.SetLogicalBounds(2, 2);
    ctx.Check(m.Rows() == 2 && m.Cols() == 2, "SetLogicalBounds failed to shrink logical size");

    // Prove the capacity wasn't destroyed!
    m.SetLogicalBounds(4, 4);
    ctx.Check(m.Rows() == 4 && m.Cols() == 4, "SetLogicalBounds destroyed capacity during shrink");

    ctx.PrintResult();
}

void TestShrinkToFitEdgeCases() {
    TestContext ctx("TestShrinkToFitEdgeCases");
    std::cout << "Running TestShrinkToFitEdgeCases..." << std::endl;

    TinyMatrix m(2, 2, {5, 6, 7, 8});

    // Edge: ShrinkToFit on already tight matrix (no-op)
    m.ShrinkToFit();
    ctx.Check(m.Rows() == 2 && m.Cols() == 2, "ShrinkToFit on tight matrix changed size");

    // Reserve + expand + shrink cycle
    m.Reserve(10, 10);
    m.SetLogicalBounds(6, 6);
    m(5, 5, 999);
    m.ShrinkToFit();

    ctx.Check(m.Rows() == 6 && m.Cols() == 6, "ShrinkToFit after expand failed");
    ctx.CheckIntVal(m, 5, 5, 999);

    // Shrink again (should still be fine)
    m.ShrinkToFit();
    ctx.CheckIntVal(m, 5, 5, 999);

    ctx.PrintResult();
}

void TestShapeWithCapacityEdgeCases() {
    TestContext ctx("TestShapeWithCapacityEdgeCases");
    std::cout << "Running TestShapeWithCapacityEdgeCases..." << std::endl;

    TinyMatrix m(2, 2, {1, 2, 3, 4});
    m.Reserve(8, 8);

    // Shape to larger size using fast path
    m.Shape(5, 5);
    ctx.Check(m.Rows() == 5 && m.Cols() == 5, "Shape fast expansion failed");
    ctx.CheckIntVal(m, 0, 0, 1);

    // Absolute reshape (forces reallocation + tight packing)
    m.Shape(3, 3, true);
    ctx.Check(m.Rows() == 3 && m.Cols() == 3, "Absolute Shape failed");
    ctx.CheckIntVal(m, 0, 0, 1);

    // Shape back to original using fast path again
    m.Reserve(10, 10);
    m.Shape(7, 7);
    ctx.Check(m.Rows() == 7 && m.Cols() == 7, "Second fast Shape failed");

    ctx.PrintResult();
}

void TestDataIntegrityAfterCapacityChanges() {
    TestContext ctx("TestDataIntegrityAfterCapacityChanges");
    std::cout << "Running TestDataIntegrityAfterCapacityChanges..." << std::endl;

    TinyMatrix m(3, 3);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            m(i, j, i * 10 + j);

    // Stress test: multiple Reserve → Expand → Shrink cycles
    for(int cycle = 0; cycle < 5; cycle++) {
        m.Reserve(20 + cycle, 20 + cycle);
        m.SetLogicalBounds(10 + cycle, 10 + cycle);
        m(9 + cycle, 9 + cycle, 12345 + cycle);   // Write sentinel
        m.ShrinkToFit();
    }

    // Verify original data is still correct after many operations
    ctx.CheckIntVal(m, 0, 0, 0);
    ctx.CheckIntVal(m, 2, 2, 22);

    // Verify last sentinel value survived
    ctx.CheckIntVal(m, 13, 13, 12349);

    ctx.PrintResult();
}

void TestOperationsOnStridedMatrices() {
    TestContext ctx("TestOperationsOnStridedMatrices");
    std::cout << "Running TestOperationsOnStridedMatrices..." << std::endl;

    TinyMatrix m(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    m.Reserve(6, 6);
    m.SetLogicalBounds(4, 4);

    // Fill expanded area
    m(2, 2, 10.0f);
    m(3, 3, 20.0f);

    // Element-wise operations should work correctly with stride
    m.multiply(2.0f);
    ctx.CheckFloatVal(m, 0, 0, 2.0f);
    ctx.CheckFloatVal(m, 3, 3, 40.0f);

    // ReLU on strided matrix
    TinyMatrix reluTest(2, 2, {-1.0f, 5.0f, -3.0f, 0.0f});
    reluTest.Reserve(5, 5);
    reluTest.SetLogicalBounds(3, 3);
    reluTest.Relu();
    ctx.CheckFloatVal(reluTest, 0, 0, 0.0f);
    ctx.CheckFloatVal(reluTest, 0, 1, 5.0f);

    ctx.PrintResult();
}

// ==================== Main ====================

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
    TestFixedPointQ88();

    // New capacity/stride tests
    TestReserveAndExpand();
    TestShapeFastExpansion();
    TestShrinkToFit();
    TestStridedMatrixOperations();
    TestWriteReadRawWithCapacity();
    TestCopyWithStride();

    // New edge case tests
    TestReserveEdgeCases();
    TestSetLogicalBoundsEdgeCases();
    TestShrinkToFitEdgeCases();
    TestShapeWithCapacityEdgeCases();
    TestDataIntegrityAfterCapacityChanges();
    TestOperationsOnStridedMatrices();

    std::cout << "\n==========================================\n";
    if(g_primary_errors == 0) {
        std::cout << "  ALL TESTS PASSED!\n";
    } else {
        std::cout << "  TESTS COMPLETED WITH ERRORS\n";
        std::cout << "  Primary Errors : " << g_primary_errors << "\n";
        std::cout << "  Cascade Errors : " << g_cascade_errors << "\n";
    }
    std::cout << "==========================================\n";

    TinyMatrix::CleanupEngine();
    return g_primary_errors > 0 ? 1 : 0;
}