import numpy as np
import torch

def normalize(data, dim):
    #print("data",data)
    data1 = data / np.sqrt(dim)
    #print("norm_data",data1)
    return data1

def softmax(x):
    # Subtracting the max value along each row (axis=1) for stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Dividing by the row-wise sum of exponentials
    return exp_x / exp_x.sum(axis=1, keepdims=True)

if __name__ == '__main__':
    variance_vector = np.full(5,0)
    norm_variance_vector = np.full(5, 0)

    numloops = 2
    numTokens = 5
    numDimensions = 1000
    for i in range(numloops):
        data_mat_1 = np.random.rand(numTokens,numDimensions)
        data_mat_2 = np.random.rand(numDimensions,numTokens)

        #dot product
        dot_product_1_2 = np.dot(data_mat_1, data_mat_2)
        print("dotproduct_1_2:", dot_product_1_2)

        normalized_dot_product_1_2 = normalize(dot_product_1_2, numDimensions)
        #print("norm_dotproduct_1_2:", normalized_dot_product_1_2)

        #covariance matrix for variance
        cov_matrix1_2 = np.cov(dot_product_1_2, ddof=0)
        normalized_cov_matrix1_2 = np.cov(normalized_dot_product_1_2, ddof=0)

        #softmax
        softmax_1_2 = softmax(dot_product_1_2)
        #print("softmax_1_2:", softmax_1_2)

        norm_softmax_1_2 = softmax(normalized_dot_product_1_2)
        #print("norm_softmax_1_2:", norm_softmax_1_2)

        #variance extraction
        variance_matrix1_2 = np.diag(cov_matrix1_2)
        #print("variance_matrix_1_2:", variance_matrix1_2)

        norm_variance_matrix1_2 = np.diag(normalized_cov_matrix1_2)

        variance_vector = np.add(variance_vector, variance_matrix1_2)
        #print("variance_vector:", variance_vector)

        norm_variance_vector = np.add(norm_variance_vector, norm_variance_matrix1_2)

    #print("variance_vector:", variance_vector)
    #print("variance_vector: ",variance_vector / numloops)
    print("norm_variance_vector: ", norm_variance_vector / numloops)