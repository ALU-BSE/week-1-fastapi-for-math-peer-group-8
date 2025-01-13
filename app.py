from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import math
import uvicorn

app = FastAPI()

# Define the Pydantic model for matrix input
class Matrix(BaseModel):
    matrix: List[List[int]]  # 5x5 matrix

# Sigmoid function definition
def sigmoid_function(matrix):
    return 1 / (1 + np.exp(-np.array(matrix)))

# Matrix multiplication with NumPy
def matrix_multiplication_numpy(matrix):
    return np.dot(matrix, matrix)  # Example: matrix multiplication with itself

# Matrix multiplication without NumPy
def matrix_multiply_without_numpy(matrix):
    # Basic matrix multiplication without NumPy (using nested loops)
    result = [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*matrix)]
        for row in matrix
    ]
    return result

# POST endpoint to process the matrix
@app.post("/calculate")
def func(matrix: Matrix):
    # Check if matrix is 5x5
    if len(matrix.matrix) != 5 or len(matrix.matrix[0]) != 5:
        return {"error": "Input matrix must be a 5x5 matrix."}

    # Perform calculations
    numpy_result = matrix_multiplication_numpy(matrix.matrix)  # Using NumPy
    non_numpy_result = matrix_multiply_without_numpy(matrix.matrix)  # Without NumPy
    sigmoid_result = sigmoid_function(non_numpy_result)  # Apply sigmoid to result

    # Return results
    return {
        "matrix_multiplication": numpy_result.tolist(),
        "non_numpy_multiplication": non_numpy_result,
        "sigmoid_output": sigmoid_result.tolist(),
    }

if __name__ == "__main__":
    uvicorn.run(app)