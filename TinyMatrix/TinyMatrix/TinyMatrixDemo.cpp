#include "TinyMatrix.h"

template<typename T>
constexpr auto int16_rand(T gen) { return ((rand() % gen) - (gen/4)); } 



int main()
{
    TinyMatrix myMatrix3(1, 3, 0);  //1 row, 3 columns, indexed from 0
    TinyMatrix myMatrix4(3, 1, 1);  //3 rows 1 columns, indexed from 1
    TinyMatrix myMatrix1(3, 2, 1);
    TinyMatrix myMatrix5(3, 3, 0);
    TinyMatrix myMatrix6(2, 2, 1);
    TinyMatrix myMatrix7(2, 4, 0);
    TinyMatrix myMatrix8(4, 2, 0);

    srand(0xBA550B0E);
    //srand((unsigned int)((unsigned long long int)((void*)&main)) + clock());
    
    //set a value of matrix (row,col,value)
    myMatrix7(0, 0, 3.1415927f); 
    printf("Get a Value:%f\n", (float)myMatrix7(0, 0));
    for (int r = 0; r < 8; r++) {
        float fr = (float)((double)rand() / (double)(RAND_MAX) * 2) - 1.0f;
        float fr2 = (float)((double)rand() / (double)(RAND_MAX) * 2) - 1.0f;
        myMatrix7(r / 4, r % 4, fr);
        myMatrix8(r % 4, r / 4, fr2);
    }

    

    int16_t trand = 0;

    
    //nasty loop to initialize matricies
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if ((i < myMatrix3.Rows() || i < myMatrix4.Rows() || i < myMatrix1.Rows() 
                   || i < myMatrix5.Rows() || i < myMatrix6.Rows() || i < myMatrix7.Rows()) &&
                (j < myMatrix3.Cols() || j < myMatrix4.Cols() || j < myMatrix1.Cols() 
                   || j < myMatrix5.Cols() || j < myMatrix6.Cols() || j < myMatrix7.Cols())) {

                if (i < myMatrix3.Rows() && j < myMatrix3.Cols()) {
                    trand = int16_rand(20);
                    myMatrix3(i + myMatrix3.IndexMode(), j + myMatrix3.IndexMode(), trand);
                }
                if (i < myMatrix4.Rows() && j < myMatrix4.Cols()) {
                    trand = int16_rand(20);
                    myMatrix4(i + myMatrix4.IndexMode(), j + myMatrix4.IndexMode(), trand);
                }
                if (i < myMatrix1.Rows() && j < myMatrix1.Cols()) {
                    trand = int16_rand(20);
                    myMatrix1(i + myMatrix1.IndexMode(), j + myMatrix1.IndexMode(), trand);
                }
                if (i < myMatrix5.Rows() && j < myMatrix5.Cols()) {
                    trand = int16_rand(20);
                    myMatrix5(i + myMatrix5.IndexMode(), j + myMatrix5.IndexMode(), trand);
                }
                if (i < myMatrix6.Rows() && j < myMatrix6.Cols()) {
                    trand = int16_rand(20);
                    myMatrix6(i + myMatrix6.IndexMode(), j + myMatrix6.IndexMode(), trand);
                }
            }
            else {
                break;
            }
        }
    }

    TinyMatrix myMatrix2(myMatrix1); //2 is a copy of 1

    printf("M1\n");
    myMatrix1.print("\n"); //prints matrix + any string provided

    printf("M1(reshape)\n");
    myMatrix1.Shape(2, 3).print("\n");

    printf("M2\n");
    myMatrix2.print("\n");
   
    printf("M2 dot M1\n");
    //make a matrix to store answer of dot product
    TinyMatrix tDota(myMatrix2.Rows(), myMatrix1.Cols(), 0);
    tDota.dot(myMatrix2, myMatrix1).print("\n");

    //result matrix will size itself also demo pointer working...
    printf("M2 dot M1 (self sized)\n");
    TinyMatrix* tDot = new TinyMatrix;
    tDot->dot(myMatrix2, myMatrix1).print("\n");

    //other matricies for transpose/dot intro
    printf("M3\n");
    myMatrix3.print("\n"); 

    printf("M4\n");
    myMatrix4.print("\n");
    
    //you can transpose a matrix manually, if matricies are compatible .dot will auto transpose second parameter
    printf("M3 dot M4 (resize, transpose manually)\n");
    tDot->dot(myMatrix3, myMatrix4.transpose()).print("\n");

    delete tDot;

    //change it back manually, manual transpose modifies in place;
    myMatrix4.transpose();
    
    //auto transpose, this will return dot product of a matrix and its transpose
    printf("M4 dot M4\n");
    tDota.dot(myMatrix4).print("\n");
    
    //you can change the shape of a matrix and its data structure will resize.
    //(default relative positioning, values are shifted to fit size, extra space will be 0 filled,
    //resizing to a smaller size will truncate values outside of the range).
    //specify 3rd optional parameter as true to use absolute positioning...(covered later)
    printf("M3 dot M4\n");
    tDota.Shape(myMatrix3.Rows(), myMatrix4.Cols()).dot(myMatrix3, myMatrix4).print("\n");
    
    //setting up a 3x3 matrix to show scalars and matrix +,
    printf("M4 dot M3\n");
    tDota.dot(myMatrix4, myMatrix3).print("\n");
    
    //add by scalar
    printf("Previous+1\n");
    tDota.add(1).print("\n");
    
    //get sum of an int16_t array
    int32_t tDot_sum = tDota.sum();
    printf("SUM:\n%ld\n\n", tDot_sum);

    //half float matrix being used in more scalar operations (int16_t is also supported)
    printf("M7\n");
    myMatrix7.print("\n");

    //get sum of a half float array (2 byte float implimentation ~3.3 decimal precision)
    float myMatrix7_sum = myMatrix7.sum();
    printf("M7 SUM:\n%.3f\n\n", myMatrix7_sum);

    //variatic declaration, {Rows,Cols,index mode[0,1],floats....}
    TinyMatrix m8(4, 2, 0, -1.712, -0.412, 0.507, 0.990, 1.309, 0.482, -0.758, -0.498 );
    printf("variatic float creation...\n");
    m8.print("\n");

    //variatic declaration, {Rows,Cols,index mode[0,1],ints....}
    TinyMatrix m9(3, 3, 0, 88, 23, 34, 76, 2, 34, 44, 92, 12);
    printf("variatic int creation...\n");
    m9.print("\n");

    //multiply by a scalar
    printf("M7*2\n");
    myMatrix7.multiply(2).print("\n");
    
    //subtract by a scalar
    printf("M7-5\n");
    myMatrix7.sub(5).print("\n");

    //addition by a scalar
    printf("M7+0.12\n");
    myMatrix7.add(0.12).print("\n");
    
    //displaying additional matrix to be used in the next operations
    printf("M8\n");
    myMatrix8.print("\n");

    //add matricies auto re-shape 2nd param true to auto reshape.. 
    //this does a relative reshape of passed in matrix without altering original
    printf("M7+M8\n");
    myMatrix7.add(myMatrix8,true).print("\n");

    //same thing... but alters original via manual transpose
    printf("M7+M8\n");
    myMatrix7.add(myMatrix8.transpose()).print("\n");

    //same thing... alters originals
    printf("M8+M7\n");
    myMatrix8.transpose().add(myMatrix7.transpose()).print("\n");

    //subtract matricies of same shape above applies
    printf("M7 - M8\n");
    myMatrix7.sub(myMatrix8).print("\n");


    //used above, printing for use ahead...
    printf("tDota\n");
    tDota.print("\n");

    //manually reshape float matrix to match shape of tDota int matrix and add them. 
    //absolute (3rd parameter 'true') maps row/col positions 1:1 0 fill any that dont map within new shape.
    printf("absolute reshaped M7\n");
    myMatrix7.Shape(tDota.Rows(), tDota.Cols(),true).print("\n");

    //add floats to converts int16_t matrix to half float matrix
    printf("M7 + tDota\n");
    myMatrix7.add(tDota).print("\n");

    printf("tDota\n");
    tDota.print("\n");
    //reshape float matrix to match shape of tDota int matrix and add them.
    //math operations on ints with half floats will convert to half float...
    //relative (no or false 3rd parameter) maps data positions as they are stored.
    //wrapping to match shape.. 0 fill extra space, truncate missing space.
    printf("relative reshaped M7\n");
    myMatrix7.Shape(tDota.Rows(), tDota.Cols()).print("\n");
    printf("tDota - relative reshaped M7\n");
    tDota.sub(myMatrix7).print("\n");

    //add ints to floats, remains a half float matrix however....
    //.Ints() converts a half float matrix to an int16_t matrix
     printf("tDota + M7\n");
    tDota.add(myMatrix7.Ints()).print("\n");

    //showing that .Ints converted this back to an int16_t matrix
    printf("M7 after .Ints()\n");
    myMatrix7.print("\n");

    //scalar float multiplication with int matrix
    printf("M7 * 0.5\n");
    myMatrix7.multiply(0.5).print("\n");

    //scalar float multiplication with float matrix
    printf("M7 * -2.25\n");
    myMatrix7.multiply(-2.25).print("\n");

    //different reshape demos.... relative/absolute, auto/manual... etc.
    printf("M5\n");
    myMatrix5.print("\n");

    printf("M6\n");
    myMatrix6.print("\n");

    printf("M5 + M6 (auto absolute reshape)\n");
    myMatrix5.add(myMatrix6, true).print("\n");

    printf("M6 after(unmodified by auto reshape):\n");
    myMatrix6.print("\n");

    printf("M6 relative reshape larger:\n");
    myMatrix6.Shape(myMatrix6.Rows() + 2, myMatrix6.Cols() + 2).print("\n");

    printf("M6 after(modified by manual reshape):\n");
    myMatrix6.print("\n");

    printf("M6 relative reshape smaller:\n");
    myMatrix6.Shape(myMatrix6.Rows()-2, myMatrix6.Cols()-2).print("\n");

    printf("M6 absolute reshape larger:\n");
    myMatrix6.Shape(myMatrix6.Rows() + 2, myMatrix6.Cols() + 2,true).print("\n");

    printf("M6 absolute reshape smaller:\n");
    myMatrix6.Shape(myMatrix6.Rows()-2, myMatrix6.Cols()-2,true).print("\n");


    
     return 0;
}
