import numpy as np
from sklearn.decomposition import NMF
import re

# Make square 
def make_square(data):
    # Split the CSV data into rows
    rows = data.strip().split("\n")
    # Convert rows into a list of lists (2D array)
    matrix = [list(map(lambda x: int(float(x)), re.split(r'[,\s]+', row.strip()))) for row in rows]    
    # matrix = [list(map(int, row.strip().split(","))) for row in rows]

    min_dimension = min(len(matrix), len(matrix[0]))
    return [row[:min_dimension] for row in matrix[:min_dimension]]

# Perform NMF
def perform_nmf(V, rank):
    model = NMF(n_components=rank, init='random', random_state=42, max_iter=2000)
    W = model.fit_transform(V)
    H = model.components_
    return W, H, V - np.dot(W, H)  

def estimate_probabilities(matrix, bin_width):
    min_val, max_val = np.min(matrix), np.max(matrix)
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    hist, bin_edges = np.histogram(matrix, bins=bins, density=True)
    return hist, bin_edges

def calculate_description_length(matrix, hist, bin_edges):
    dl = 0
    for i in range(len(hist)):
        if hist[i] > 0:
            p = hist[i] * (bin_edges[i+1] - bin_edges[i])  # Probability of bin
            dl -= p * np.log2(p)  # entropy
    return dl * np.prod(matrix.shape)

def find_optimal_rank(V, max_rank, bin_width):
    optimal_rank = None
    min_description_length = np.inf

    for r in range(1, max_rank + 1):
        W, H, E = perform_nmf(V, r)
        
        # Estimate probabilities and calculate DL for W, H, and E
        hist_W, bins_W = estimate_probabilities(W, bin_width)
        hist_H, bins_H = estimate_probabilities(H, bin_width)
        hist_E, bins_E = estimate_probabilities(E, bin_width)
        
        dl_W = calculate_description_length(W, hist_W, bins_W)
        dl_H = calculate_description_length(H, hist_H, bins_H)
        dl_E = calculate_description_length(E, hist_E, bins_E)
        
        total_dl = dl_W + dl_H + dl_E
        print(f"Rank {r}, Description Length: {total_dl}")

        if total_dl < min_description_length:
            min_description_length = total_dl
            optimal_rank = r

    return optimal_rank

with open('SCMFDD_matrix', 'r') as data:

    file_content = data.read()
    matrix = make_square(file_content)

optimal_rank = find_optimal_rank(matrix, 30, 0.01)  # Adjust bin width as needed
print(f"Optimal rank is {optimal_rank}")