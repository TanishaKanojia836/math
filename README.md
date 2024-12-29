# math
# 1) import numpy as np

NR = int(input("Enter the number of rows: "))
NC = int(input("Enter the number of columns: "))

print("Enter the entries in a single line (separated by space): ")

entries = list(map(int, input().split()))

A  = np.array(entries).reshape(NR, NC)

print("Matrix A is as follows:", "\n", A)


# For finding the inverse of a Matrix
A_Inverse = np.linalg.inv(A)

print("Inverse of A is", "\n", A_Inverse)


#For finding the transpose of a Matrix
Transpose_of_A_Inverse = np.transpose(A_Inverse)

print("Transpose of A Inverse is", "\n", Transpose_of_A_Inverse)


# For finding the determinant of a Matrix
Determinant_of_A= np.linalg.det(A)

print("Determinant of A is", "\n", Determinant_of_A)


# For finding the cofactor of a Matrix
Cofactor_of_A = np.dot(Transpose_of_A_Inverse, Determinant_of_A)

print("The Cofactor of a Matrix is:", "\n", Cofactor_of_A)


# For finding the Adjoint of a Matrix
Adjoint_of_A = np.transpose(Cofactor_of_A)

print("The Adjoint of a Matrix is:", "\n", Adjoint_of_A)
# 2) import numpy as np

#Taking Input of Matrix From user (User Define matrix Input)

#NR: Number of RoWS

#NC: Number of Column

NR = int(input("Enter the number of rows: "))

NC = int(input("Enter the number of columns: "))

print("Enter the entries in a single line (separated by space): ")

#User input of entries in a

#single line separated by space

entries = list(map(int, input().split()))

#For printing the matrix

matrix = np.array(entries).reshape(NR, NC)

print("Matrix X is as follows:", "\n", matrix)

#For finding the Rank of a Matrix

print("The Rank of a Matrix is:", np.linalg.matrix_rank(matrix))
# 3) import numpy as np

##Taking Input of Matrix From user (User Define matrix Input)

##NR: Number of Rows

##NC: Number of Column

###Coefficient Matrix (A) Elements

print("Enter the dimension of coefficients matrix (A):")

NR = int(input("Enter the number of rows: "))

NC = int(input("Enter the number of columns: "))

print("Enter the elements of coefficients matrix (A) in a single line (separated by space):")

Coefficients_Entries = list(map(float, input().split()))

Coefficient_Matrix = np.array(Coefficients_Entries).reshape(NR, NC)

print("Coefficient Matrix (A) is as follows:", "\n",

Coefficient_Matrix, "\n")

###Column Matrix (B) Elements

print("Enter the elements of column matrix (B) in a single line (separated by space):")

Column_Entries = list(map(float, input().split()))

Column_Matrix = np.array(Column_Entries).reshape(NR, 1)

print("Column Matrix (B) is as follows:", "\n", Column_Matrix, "\n")

# #Solution of Homogeneous System of Equations using Gauss elimination method

Solution_of_the_system_of_Equations = np.linalg.solve(Coefficient_Matrix, Column_Matrix)

print("Solution of the system of Equations using Gauss elimination method")

print(Solution_of_the_system_of_Equations)
# 4) import numpy as np

NR= int(input("Enter the number of rows:"))
NC= int(input("Enter the number of columns:"))
print("Enter the elements of the coefficient matrix A :")

Coefficients_Entries = list(map(float , input().split()))
Coefficient_Matrix = np.array(Coefficients_Entries).reshape(NR,NC)
print("Coefficient Matrix (A) is as follows:\n", Coefficient_Matrix, "\n")

print("Enter the number of elemnets of the column matrix (B) in a single line (seperated by spaces):")
Column_Entries = list(map(float, input().split()))
Column_Matrix = np.array(Column_Entries).reshape(NR,1)
print("Column Matrix (B) is as follows:\n", Column_Matrix,"\n")

"Solution of homogeneous system of equations using Gauss Jordan"
inv_of_coefficient_matrix = np.linalg.inv(Coefficient_Matrix)
solution_of_the_system_of_equations = np.matmul(inv_of_coefficient_matrix, Column_Matrix)
print("Solution of the system of eqations using Gauss Jordan:")
print(solution_of_the_system_of_equations)
# 5) import numpy as np

NR= int(input("Enter the number of rows:"))
NC= int(input("Enter the number of columns:"))
print("Enter the elements of the coefficient matrix A :")

Coefficients_Entries = list(map(float , input().split()))
Coefficient_Matrix = np.array(Coefficients_Entries).reshape(NR,NC)
print("Coefficient Matrix (A) is as follows:\n", Coefficient_Matrix, "\n")

print("Enter the number of elemnets of the column matrix (B) in a single line (seperated by spaces):")
Column_Entries = list(map(float, input().split()))
Column_Matrix = np.array(Column_Entries).reshape(NR,1)
print("Column Matrix (B) is as follows:\n", Column_Matrix,"\n")

"Solution of homogeneous system of equations using Gauss Jordan"
inv_of_coefficient_matrix = np.linalg.inv(Coefficient_Matrix)
solution_of_the_system_of_equations = np.matmul(inv_of_coefficient_matrix, Column_Matrix)
print("Solution of the system of eqations using Gauss Jordan:")
print(solution_of_the_system_of_equations)
# 6) import numpy as np
from scipy.linalg import eig
from sympy import Matrix, symbols, det

# Function to check diagonalizability
def is_diagonalizable(A):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    
    # Check the geometric multiplicity (number of independent eigenvectors)
    # If the number of independent eigenvectors matches the size of the matrix, it's diagonalizable
    rank = np.linalg.matrix_rank(eigenvectors)
    if rank == A.shape[0]:
        return True, eigenvalues
    else:
        return False, eigenvalues

# Function to verify the Cayley-Hamilton theorem
def verify_cayley_hamilton(A):
    # Convert matrix to sympy Matrix
    A_sympy = Matrix(A)
    
    # Compute the characteristic polynomial
    lambda_symbol = symbols('lambda')
    char_poly = A_sympy.charpoly(lambda_symbol)
    
    # The characteristic polynomial as a sympy expression
    characteristic_polynomial = char_poly.as_expr()

    # Replace lambda with the matrix A in the characteristic polynomial
    A_substitution = characteristic_polynomial.subs(lambda_symbol, A_sympy)
    
    # Check if the result is the zero matrix (Cayley-Hamilton should hold)
    return A_substitution.is_zero

# Example matrix
A = np.array([[4, -1, 1],
              [-1, 4, -2],
              [1, -2, 3]])

# Step 1: Check if the matrix is diagonalizable
diagonalizable, eigenvalues = is_diagonalizable(A)
print("Is the matrix diagonalizable?", diagonalizable)
print("Eigenvalues of the matrix:", eigenvalues)

# Step 2: Verify Cayley-Hamilton theorem
is_cayley_hamilton_true = verify_cayley_hamilton(A)
print("Does the matrix satisfy the Cayley-Hamilton theorem?", is_cayley_hamilton_true)
# 7) import sympy as sp

# Define symbols (coordinates)
x, y, z = sp.symbols('x y z')

# Example Scalar Field f(x, y, z)
f = x*2 + y2 + z*2

# Example Vector Field A(x, y, z)
A_x = x * y
A_y = y * z
A_z = z * x
A = sp.Matrix([A_x, A_y, A_z])

# 1. Compute the Gradient of the scalar field f
gradient_f = sp.Matrix([sp.diff(f, var) for var in (x, y, z)])
print("Gradient of f(x, y, z):")
sp.pprint(gradient_f)

# 2. Compute the Divergence of the vector field A
divergence_A = sp.diff(A_x, x) + sp.diff(A_y, y) + sp.diff(A_z, z)
print("\nDivergence of A(x, y, z):")
sp.pprint(divergence_A)

# 3. Compute the Curl of the vector field A
curl_A = sp.Matrix([
    sp.diff(A_z, y) - sp.diff(A_y, z),  # i-component
    sp.diff(A_x, z) - sp.diff(A_z, x),  # j-component
    sp.diff(A_y, x) - sp.diff(A_x, y)   # k-component
])
print("\nCurl of A(x, y, z):")
sp.pprint(curl_A)
