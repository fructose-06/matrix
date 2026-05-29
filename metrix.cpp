#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

const double EPSILON = 1e-5; 
const double PI = acos(-1.0);

// Function to print a matrix beautifully
void printMatrix(const string& name, const vector<vector<double>>& matrix) {
    cout << "\n--- " << name << " ---\n";
    for(const auto& row : matrix) {
        for(double val : row) {
            if(abs(val) < EPSILON) val = 0.0; // Prevent printing annoying -0.0000
            cout << setw(10) << setprecision(4) << val << " "; 
        }
        cout << endl;
    }
}

// Function for matrix multiplication (C = A * B)
vector<vector<double>> multiplyMatrix(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Function to check if a matrix is truly diagonal
bool isDiagonal(const vector<vector<double>>& M) {
    int n = M.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && abs(M[i][j]) > EPSILON) {
                return false; 
            }
        }
    }
    return true;
}

// Function to find linearly independent eigenvectors using Gaussian Elimination & Back-substitution
vector<vector<double>> solveNullSpace(vector<vector<double>> M) {
    int n = M.size();
    
    // Step 1: Convert to Row Echelon Form (Upper Triangular) using Partial Pivoting
    for (int i = 0; i < n; i++) {
        int max_r = i;
        for (int r = i + 1; r < n; r++) {
            if (abs(M[r][i]) > abs(M[max_r][i])) max_r = r;
        }
        if (abs(M[max_r][i]) > EPSILON) {
            swap(M[i], M[max_r]);
            for (int r = i + 1; r < n; r++) {
                double factor = M[r][i] / M[i][i];
                for (int c = i; c < n; c++) {
                    M[r][c] -= factor * M[i][c];
                }
            }
        }
    }

    // Step 2: Identify Free Variables based on the diagonal elements
    vector<bool> is_free(n, false);
    int free_count = 0;
    for (int i = 0; i < n; i++) {
        if (abs(M[i][i]) < EPSILON) {
            is_free[i] = true;
            free_count++;
        }
    }

    vector<vector<double>> basis;
    if (free_count == 0) return basis; 

    // Step 3: Back-substitution for each Free Variable to build basis vectors
    for (int f = 0; f < n; f++) {
        if (!is_free[f]) continue;

        vector<double> vec(n, 0.0);
        vec[f] = 1.0; 

        for (int i = n - 1; i >= 0; i--) {
            if (is_free[i]) continue;
            
            int pivot_col = -1;
            for (int c = 0; c < n; c++) {
                if (abs(M[i][c]) > EPSILON) {
                    pivot_col = c;
                    break;
                }
            }
            
            if (pivot_col != -1) {
                double sum = 0;
                for (int c = pivot_col + 1; c < n; c++) {
                    sum += M[i][c] * vec[c];
                }
                vec[pivot_col] = -sum / M[i][pivot_col];
            }
        }

        // Normalize the vector to length 1.0
        double norm = 0;
        for (double val : vec) norm += val * val;
        norm = sqrt(norm);
        if (norm > EPSILON) {
            for (double &val : vec) val /= norm;
            basis.push_back(vec);
        }
    }
    return basis;
}

// Solver for 2x2 matrices
void solve2x2(vector<vector<double>> A) {
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];

    double trace = a + d;
    double det = (a * d) - (b * c);
    double discriminant = (trace * trace) - (4 * det);

    if (discriminant < -EPSILON) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Eigenvalues are complex numbers\n";
        return;
    }

    double L1 = (trace + sqrt(max(0.0, discriminant))) / 2.0;
    double L2 = (trace - sqrt(max(0.0, discriminant))) / 2.0;

    cout << "\n1. Eigenvalues:\nL1 = " << L1 << ", L2 = " << L2 << "\n";

    vector<vector<double>> P_cols;
    // FIXED: If eigenvalues are identical, solve null space once to get all independent vectors
    if (abs(L1 - L2) < 1e-3) {
        vector<vector<double>> M = {{a - L1, b}, {c, d - L1}};
        P_cols = solveNullSpace(M);
    } else {
        for (double L : {L1, L2}) {
            vector<vector<double>> M = {{a - L, b}, {c, d - L}};
            auto vecs = solveNullSpace(M);
            for (auto& v : vecs) P_cols.push_back(v);
        }
    }

    if (P_cols.size() < 2) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Deficient eigenvectors (Geometric multiplicity < Algebraic multiplicity)\n";
        return;
    }

    vector<vector<double>> P = { {P_cols[0][0], P_cols[1][0]}, {P_cols[0][1], P_cols[1][1]} };
    double detP = (P[0][0] * P[1][1]) - (P[0][1] * P[1][0]);

    if (abs(detP) < EPSILON) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Eigenvectors are linearly dependent (det(P) = 0)\n";
        return;
    }

    vector<vector<double>> P_inv = {
        {P[1][1] / detP, -P[0][1] / detP},
        {-P[1][0] / detP, P[0][0] / detP}
    };

    // FIXED: Strict Final Mathematical Validation
    vector<vector<double>> D = multiplyMatrix(multiplyMatrix(P_inv, A), P);
    if (!isDiagonal(D)) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Matrix D is not truly diagonal (Deficient eigenvectors detected via validation)\n";
        return;
    }

    cout << "\n2. Eigenvectors:\n";
    cout << "v1 = [" << P[0][0] << ", " << P[1][0] << "]\n";
    cout << "v2 = [" << P[0][1] << ", " << P[1][1] << "]\n";
    cout << "\n3. Diagonalization Check: Can be diagonalized\n";

    cout << "\n4. Matrices P and P^-1:";
    printMatrix("Matrix P", P);
    printMatrix("Matrix P^-1", P_inv);
    printMatrix("Diagonal Matrix D (P^-1 * A * P)", D);
}

// Solver for 3x3 matrices
void solve3x3(vector<vector<double>> A) {
    double c2 = -(A[0][0] + A[1][1] + A[2][2]);
    double c1 = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) + 
                (A[0][0]*A[2][2] - A[0][2]*A[2][0]) + 
                (A[1][1]*A[2][2] - A[1][2]*A[2][1]);
    double c0 = -(A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) - 
                  A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) + 
                  A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]));

    double Q = (3 * c1 - c2 * c2) / 9.0;
    double R = (9 * c2 * c1 - 27 * c0 - 2 * c2 * c2 * c2) / 54.0;
    double D_val = Q * Q * Q + R * R;

    vector<double> raw_eigenvalues;
    if (D_val > EPSILON) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Complex eigenvalues encountered\n";
        return;
    } else {
        double theta = acos(max(-1.0, min(1.0, R / sqrt(-Q * Q * Q))));
        raw_eigenvalues.push_back(2 * sqrt(-Q) * cos(theta / 3.0) - c2 / 3.0);
        raw_eigenvalues.push_back(2 * sqrt(-Q) * cos((theta + 2 * PI) / 3.0) - c2 / 3.0);
        raw_eigenvalues.push_back(2 * sqrt(-Q) * cos((theta + 4 * PI) / 3.0) - c2 / 3.0);
    }

    vector<pair<double, int>> unique_eigenvalues;
    for (double ev : raw_eigenvalues) {
        bool found = false;
        for (auto& u : unique_eigenvalues) {
            if (abs(u.first - ev) < 1e-3) {
                u.second++;
                found = true;
                break;
            }
        }
        if (!found) unique_eigenvalues.push_back({ev, 1});
    }

    cout << "\n1. Eigenvalues:\n";
    int idx = 1;
    for(double ev : raw_eigenvalues) cout << "L" << idx++ << " = " << ev << "\n";

    vector<vector<double>> P_cols;
    for (auto& pair : unique_eigenvalues) {
        double L = pair.first;
        vector<vector<double>> M = A;
        for (int i = 0; i < 3; i++) M[i][i] -= L;

        vector<vector<double>> vecs = solveNullSpace(M);
        for (auto& v : vecs) P_cols.push_back(v);
    }

    if (P_cols.size() < 3) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Geometric multiplicity is less than algebraic multiplicity (not enough unique eigenvectors)\n";
        return;
    }

    vector<vector<double>> P(3, vector<double>(3));
    for (int i = 0; i < 3; i++) {
        P[0][i] = P_cols[i][0]; P[1][i] = P_cols[i][1]; P[2][i] = P_cols[i][2];
    }

    double detP = P[0][0]*(P[1][1]*P[2][2] - P[1][2]*P[2][1]) - 
                  P[0][1]*(P[1][0]*P[2][2] - P[1][2]*P[2][0]) + 
                  P[0][2]*(P[1][0]*P[2][1] - P[1][1]*P[2][0]);

    if (abs(detP) < EPSILON) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Eigenvectors are not linearly independent (det(P) = 0)\n";
        return;
    }

    vector<vector<double>> P_inv(3, vector<double>(3));
    P_inv[0][0] = (P[1][1]*P[2][2] - P[1][2]*P[2][1]) / detP;
    P_inv[0][1] = (P[0][2]*P[2][1] - P[0][1]*P[2][2]) / detP;
    P_inv[0][2] = (P[0][1]*P[1][2] - P[0][2]*P[1][1]) / detP;
    P_inv[1][0] = (P[1][2]*P[2][0] - P[1][0]*P[2][2]) / detP;
    P_inv[1][1] = (P[0][0]*P[2][2] - P[0][2]*P[2][0]) / detP;
    P_inv[1][2] = (P[0][2]*P[1][0] - P[0][0]*P[1][2]) / detP;
    P_inv[2][0] = (P[1][0]*P[2][1] - P[1][1]*P[2][0]) / detP;
    P_inv[2][1] = (P[0][1]*P[2][0] - P[0][0]*P[2][1]) / detP;
    P_inv[2][2] = (P[0][0]*P[1][1] - P[0][1]*P[1][0]) / detP;

    // FIXED: Strict Final Mathematical Validation
    vector<vector<double>> D = multiplyMatrix(multiplyMatrix(P_inv, A), P);
    if (!isDiagonal(D)) {
        cout << "\n3. Diagonalization Check: Cannot be diagonalized\n";
        cout << "Reason: Matrix D is not truly diagonal (Deficient eigenvectors detected via validation)\n";
        return;
    }

    cout << "\n2. Eigenvectors:\n";
    for (int i = 0; i < 3; i++) {
        cout << "v" << i+1 << " = [" << P_cols[i][0] << ", " << P_cols[i][1] << ", " << P_cols[i][2] << "]\n";
    }
    cout << "\n3. Diagonalization Check: Can be diagonalized\n";

    cout << "\n4. Matrices P and P^-1:";
    printMatrix("Matrix P", P);
    printMatrix("Matrix P^-1", P_inv);
    printMatrix("Diagonal Matrix D (P^-1 * A * P)", D);
}

int main() {
    int dim;
    cout << "Input dimension of matrix (2 or 3): ";
    cin >> dim;

    if (dim != 2 && dim != 3) {
        cout << "Error: Only 2x2 or 3x3 matrices are supported.\n";
        return 0;
    }

    vector<vector<double>> matrix(dim, vector<double>(dim));
    cout << "Input matrix values row by row:\n";
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            cin >> matrix[i][j];
        }
    }

    printMatrix("Input Matrix A", matrix);

    if (dim == 2) solve2x2(matrix);
    else solve3x3(matrix);

    return 0;
}