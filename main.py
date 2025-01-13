from fastapi import FastAPI
import uvicorn 
from pydantic import BaseModel
import numpy as np
import math

app = FastAPI()


@app.get("/")

def sigmoid_fuction(x):
    return  1 / (1 + math.exp(-x)) 

print(sigmoid_fuction(3))


M = np.array([[3,4,5], [6,2,1]])
B = np.array([[9,3,6], [4,7,2]])
X= np.array([[4], [5], [7]])



def matrix_multiplication_numpy():
    return np.dot(M, X) + B

print(matrix_multiplication_numpy())


def matrix_multiply_without_numpy(M, X):
    result = [[0 for _ in range(len(X[0]))] for _ in range(len(M))]
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                result[i][j] += M[i][k] * X[k][j]
    return result

print(matrix_multiply_without_numpy(M, X))


@app.post("/calculate")
def calculate(M, X, B):
    if len(M, X, B) != 5 or len((M, X, B)[0]) != 5:
        return {"error": "Input matrix must be a 5x5 matrix."}

    # Compute using NumPy
    numpy_result = matrix_multiplication_numpy(M, X, B)


if __name__ == "__main__": 
   uvicorn.run(app)