#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <random>

using namespace std;
using matrix = vector<vector<double>>;

matrix x;
matrix y;
matrix wh;
matrix bh;
matrix wout;
matrix bout;
double totalError;

// Creates a n x m matrix. If rndm is true,
// it fills the matrix with random numbers. [0, 1]
matrix createMatrix(int n, int m, bool rndm)
{
    matrix _new = matrix(n, vector<double>(m));
    for(int i = 0; i < _new.size(); i++)
        for(int j = 0; j < _new[0].size(); j++)
            _new[i][j] = ((double) rand() / RAND_MAX);

    return _new;
}

void printMatrix(matrix m, string name)
{
    cout << "--- " << name << " ---" << endl;

    for(int i = 0; i < m.size(); i++)
    {
        for(int j = 0; j < m[0].size(); j++)
            cout << m[i][j] << "\t";
        cout << endl;
    }

    cout << endl;
}

void getAnswer(matrix input, matrix pred)
{
    cout << "input = [ ";
    for(int i = 0; i < input[0].size(); i++)
        cout << ((int) round(input[0][i])) << " ";
    cout << "]" << endl;

    cout << "prediction = [ ";
    for(int i = 0; i < pred[0].size(); i++)
        cout << ((int) round(pred[0][i])) << " ";
    cout << "]" << endl;

    int mx = 0;
    for(int i = 0; i < pred[0].size(); i++)
    {
        if(pred[0][i] > pred[0][mx])
            mx = i;
    }

    cout << "Answer = " << mx << endl;
}

// Derivative of ErrorSum ->  (out - target)
matrix derivativesErrorSum(matrix target, matrix out)
{
    if(target.size() != out.size() || target[0].size() != out[0].size())
    {
        cerr << "Error Etotal() -- matrix sizes are not equal" << endl;
        exit(1);
    }

    matrix d_es = createMatrix(out.size(), out[0].size(), false);

    for(int i = 0; i < d_es.size(); i++)
        for(int j = 0; j < d_es[0].size(); j++)
            d_es[i][j] = out[i][j] - target[i][j];

    return d_es;
}

// Derivative of sigmoid function ->  sig(z) * (1 - sig(z))
matrix derivativesSigmoid(matrix x)
{
    matrix d_sig = createMatrix(x.size(), x[0].size(), false);

    for(int i = 0; i < x.size(); i++)
        for(int j = 0; j < x[0].size(); j++)
            d_sig[i][j] = x[i][j] * (1 - x[i][j]);

    return d_sig;
}

// Net function ->  z(W, b) = Sum(Wn * in) + b
// Derivative of Net w.r.t Wn ->  in
// in -> n-th input
matrix derivativesNet(matrix x)
{
    matrix d_net = createMatrix(x.size(), x[0].size(), false);

    for(int i = 0; i < x.size(); i++)
        for(int j = 0; j < x[0].size(); j++)
            d_net[i][j] = x[i][j];

    return d_net;
}

// Sigmoid function ->  sig(z) = (1 / (1 + exp(-z)))

matrix sigmoid(matrix z)
{
    matrix sigZ = createMatrix(z.size(), z[0].size(), false);

    for(int i = 0; i < z.size(); i++)
        for(int j = 0; j < z[0].size(); j++)
            sigZ[i][j] = 1 / (1 + exp(-z[i][j]));

    return sigZ;
}

// ErrorSum function ->  ((1 / 2) * ((target - out) ^ 2))// Calculates derivative of Net function
matrix calcError(matrix out, matrix y)
{
    if(out.size() != y.size() || out[0].size() != y[0].size())
    {
        cerr << "Error calcError() -- matrix sizes are not equal" << endl;
        exit(1);
    }

    matrix error = createMatrix(y.size(), y[0].size(), false);

    for(int i = 0; i < error.size(); i++)
        for(int j = 0; j < error[0].size(); j++)
            error[i][j] = ((y[i][j] - out[i][j]) * (y[i][j] - out[i][j])) / 2;

    return error;
}

// Returns dot product of two matrix.
matrix dot(matrix a, matrix b)
{
    if(a[0].size() != b.size())
    {
        cerr << "Error dot() -- can not calculate the dot product of a and b" << endl;
        exit(EXIT_FAILURE);
    }

    matrix dot_product(a.size(), vector<double>(b[0].size()));

    int i, j, k;
    for(i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b[0].size(); j++)
        {
            double sum = 0.0;
            for(k = 0; k < a[0].size(); k++)
                sum += a[i][k] * b[k][j];

            dot_product[i][j] = sum;
        }
    }

    return dot_product;
}

// Returns transpose of matrix x
matrix T(matrix x)
{
    matrix t = createMatrix(x[0].size(), x.size(), false);

    for(int i = 0; i < x.size(); i++)
        for(int j = 0; j < x[0].size(); j++)
            t[j][i] = x[i][j];

    return t;
}

// Multiply a and b, element by element
matrix mult(matrix a, matrix b)
{
    if(a.size() != b.size() || a[0].size() != b[0].size())
    {
        cerr << "Error mult() -- matrix sizes are not equal" << endl;
        exit(1);
    }

    matrix mul = createMatrix(a.size(), a[0].size(), false);

    for(int i = 0; i < mul.size(); i++)
        for(int j = 0; j < mul[0].size(); j++)
            mul[i][j] = a[i][j] * b[i][j];

    return mul;
}

// Adds a and b, element by element
matrix add(matrix a, matrix b)
{
    if(a.size() != b.size() || a[0].size() != b[0].size())
    {
        cerr << "Error add() -- matrix sizes are not equal" << endl;
        exit(1);
    }

    matrix sum = createMatrix(a.size(), a[0].size(), false);

    for(int i = 0; i < sum.size(); i++)
        for(int j = 0; j < sum[0].size(); j++)
            sum[i][j] = a[i][j] + b[i][j];

    return sum;
}

// Calculates the total error of network
double calcTotalError(matrix error)
{
    double totalError = 0.0;
    for(int i = 0; i < error.size(); i++)
        for(int j = 0; j < error[0].size(); j++)
            totalError += error[i][j];

    return totalError;
}

// Update weights w.r.t to error and learning rate
// Formula: (w - (learning rate * derivative of Error w.r.t w))
matrix update(matrix w, matrix w_d_E, double learningRate)
{
    if(w.size() != w_d_E.size() || w[0].size() != w_d_E[0].size())
    {
        cerr << "Error update() -- matrix sizes are not equal" << endl;
        exit(1);
    }

    matrix upd = createMatrix(w.size(), w[0].size(), false);

    for(int i = 0; i < upd.size(); i++)
        for(int j = 0; j < upd[0].size(); j++)
            upd[i][j] = w[i][j] - (learningRate * w_d_E[i][j]);

    return upd;
}

matrix iterate(double lr, bool queryMode, int itrCount = 1)
{
    matrix outo;

    for(int i = 0; i < itrCount; i++)
    {
        // Start of feed forward

        matrix neth = add(dot(x, wh), bh);          // calculate neth

        matrix outh = sigmoid(neth);                // calculate outh

        matrix neto = add(dot(outh, wout), bout);   // calculate neto

        outo = sigmoid(neto);                       // calculate outo

        matrix E = calcError(outo, y);              // calculateError

        totalError = calcTotalError(E);

        // If we are in query mode, don't calculate the rest of
        if(queryMode == true)
            return outo;


        // Start of backpropagation
        // Calculating the derivatives of Error w.r.t output layer weights

        matrix outo_d_E = derivativesErrorSum(y, outo);     // the partial derivative of E with respect to outo

        matrix neto_d_outo = derivativesSigmoid(outo);      // the partial derivative of outo with respect to neto

        matrix neto_d_E = mult(outo_d_E, neto_d_outo);      // the partial derivative of E with respect to neto

        matrix w_d_neto = derivativesNet(outh);             // the partial derivative of neto with respect to l1 w

        matrix wo_d_E = dot(T(neto_d_E), w_d_neto);         // the partial derivative of E with respect to l1 w

        matrix upd_wout = update(wout, T(wo_d_E), lr);      // updated weights for output layer


        // Calculating the derivatives of Error w.r.t hidden layer weights

        matrix outh_d_neto = derivativesNet(wout);          // the partial derivative of neto with respect to outh

        matrix outh_d_E = dot(outh_d_neto, T(neto_d_E));    // the partial derivative of E with respect to outh

        matrix neth_d_outh = derivativesSigmoid(outh);      // the partial derivative of outh with respect to neth

        matrix neth_d_E = mult(T(outh_d_E), neth_d_outh);   // the partial derivative of E with respect to neth

        matrix w_d_neth = derivativesNet(x);                // the partial derivative of neth with respect to l2 w

        matrix l2_w_d_E = dot(T(w_d_neth), neth_d_E);       // the partial derivative of E with respect to l2 w

        matrix upd_wh = update(wh, l2_w_d_E, lr);            // updated weights for hidden layer


        // Updates the hidden and ouput layout weights

        wh = upd_wh;
        wout = upd_wout;
    }

    return outo;
}

int main()
{
    int inputNeuronSize = 3;
    int hiddenLayerNeuronSize = 5;
    int outputNeuronSize = 8;
    double learningRate = 1;

    srand(time(NULL));

    x = createMatrix(1, inputNeuronSize, false);
    y = createMatrix(1, outputNeuronSize, false);

    // hidden layer weights
    wh = createMatrix(inputNeuronSize, hiddenLayerNeuronSize, true);

    // hidden layer biases
    bh = createMatrix(1, hiddenLayerNeuronSize, true);

    // output weights
    wout = createMatrix(hiddenLayerNeuronSize, outputNeuronSize, true);

    // output biases
    bout = createMatrix(1, outputNeuronSize, true);

    cout << "Training is started..." << endl;

    // preparing training set
    for(int i = 0; i < 1000; i++)
    {
        for(int j = 0; j <= 7; j++)
        {
            // If j = 4, we set x as 1 0 0
            // most significant bit is x[0][0]
            x[0][2] = (j & 1) ? 1 : 0;
            x[0][1] = (j & 2) ? 1 : 0;
            x[0][0] = (j & 4) ? 1 : 0;

            // At the start, all element of y is zero
            // We just need to update answer element.
            // (e.g. y[0][4] = 1, means answer is 3)
            y[0][j] = 1;

            iterate(learningRate, false, 1);

            // We reset the answer for further trainings
            y[0][j] = 0;
        }
    }

    cout << "Training is completed." << endl << endl;

    // Now we query out network.
    for(int i = 0; i <= 7; i++)
    {
        cout << "#Query " << (i + 1) << endl;
        x[0][2] = (i & 1) ? 1 : 0;
        x[0][1] = (i & 2) ? 1 : 0;
        x[0][0] = (i & 4) ? 1 : 0;

        matrix pred = iterate(learningRate, true);
        getAnswer(x, pred);

        cout << setprecision(7) << "Error = " << totalError << endl << endl;
    }

    return 0;
}
