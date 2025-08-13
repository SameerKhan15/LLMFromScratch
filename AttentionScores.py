import torch
import torch.nn.functional as F

def context_vector_dot_prod_2(inputs):
    attention_scores = inputs @ inputs.T
    attention_weights = torch.softmax(attention_scores, dim=-1)
    context_vector = attention_weights @ inputs
    return context_vector

def context_vector_dot_prod_1(inputs):
    att_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    att_weights = torch.empty(inputs.shape[0], inputs.shape[0])
    context_vectors = torch.empty(inputs.shape)

    for a, x_a in enumerate(inputs):
        for i, x_i in enumerate(inputs):
            att_scores[a][i] = torch.dot(x_i, x_a)
        att_weights[a] = torch.softmax(att_scores[a], dim=0)
        for b, x_b in enumerate(inputs):
            context_vectors[a] += att_weights[a][b] * x_b
    return context_vectors

if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89], #Your x^1
                           [0.55, 0.87, 0.66], #journey x^2
                           [0.57, 0.85, 0.64], #starts x^3
                           [0.22, 0.58, 0.33], #with x^4
                           [0.77, 0.25, 0.10], #one x^5
                           [0.05, 0.80, 0.55]]) #step x^6

    print("context_vector_dot_prod_1:",context_vector_dot_prod_1(inputs))
    print("context_vector_dot_prod_2:", context_vector_dot_prod_2(inputs))
