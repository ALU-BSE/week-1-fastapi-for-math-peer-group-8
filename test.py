from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import math
import uvicorn

app = FastAPI()

class MatrixInput(BaseModel):
    matrix: List[List[float]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Matrix multiplication function (with NumPy)
def matrix_multiply_with_numpy(M, X):
    M = np.array(M)
    X = np.array(X)
    return np.dot(M, X)

# Matrix multiplication function (without NumPy)
def matrix_multiply_without_numpy(M, X):
    result = [[0 for _ in range(len(X[0]))] for _ in range(len(M))]
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                result[i][j] += M[i][k] * X[k][j]
    return result

# Sigmoid function (element-wise)
def sigmoid(x):
    # If input is a list, apply sigmoid element-wise
    if isinstance(x, list):
        return [[1 / (1 + math.exp(-cell)) for cell in row] for row in x]
    # If input is a numpy array, apply sigmoid element-wise
    return 1 / (1 + np.exp(-x))

# Endpoint: /calculate
@app.post("/calculate")
def calculate():
    # Initialize X as a 5x1 matrix
    X = np.array([[1], [2], [3], [4], [5]])

    # Compute using NumPy
    numpy_result = matrix_operation_numpy(M, X, B)
    numpy_sigmoid_result = sigmoid(numpy_result).tolist()


# Compute without NumPy Library
    manual_result = matrix_operation_manual(M.tolist(), X.tolist(), B.tolist())
    manual_sigmoid_result = [[sigmoid(value[0])] for value in manual_result]

    return {
        "result_with_numpy": numpy_sigmoid_result,
        "result_without_numpy": manual_sigmoid_result
    }


if __name__ == "__main__":
    uvicorn.run(app)
