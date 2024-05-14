import numpy as np
import torch  
import torch.optim as optim       
import re

# Make square 
def make_square(data):
    # Split the CSV data into rows
    rows = data.strip().split("\n")
    # Assuming rows are separated by commas and can be directly converted to integers
    matrix = [list(map(lambda x: int(float(x)), re.split(r'[,\s]+', row.strip()))) for row in rows]    

    min_dimension = min(len(matrix), len(matrix[0]))
    return [row[:min_dimension] for row in matrix[:min_dimension]]

# Open the word_word_correlation data and parse the matrix
with open('SCMFDD_matrix', 'r') as data:
    file_content = data.read()
    matrix = make_square(file_content)


num_rows = len(matrix) 
num_cols = len(matrix[0]) if matrix else 0  

print("Size of matrix:", num_rows, "x", num_cols)

# NMF: torch neural network class to optimize factored matrices and W using backpropagation
class NMF(torch.nn.Module):
    # init
    # m: the original matrix containing drug-disease similarity, given as a NumPy array
    # r: the rank as an int, given in the original data
    # max_iter: max number of iterations before stopping fitting
    # tol: the loss at which to stop fitting
    def __init__(self, m: np.array, r: int, max_iter = 5000, tol = 10):
        super().__init__()
        self.M = m
        self.max_iter = max_iter
        self.tol = tol
        # Randomly initializing A and W as parameters to optimize later
        self.A = torch.nn.Parameter(torch.rand(m.shape[0], r), requires_grad=True)
        self.W = torch.nn.Parameter(torch.rand(r, m.shape[0]), requires_grad=True)
    
    # forward
    # Creates the reconstruction by multiplying A and W
    def forward(self): 
        return torch.matmul(self.A, self.W)

    # fit
    # Fits the neural net by optimizing A and W. In this case, we are not interested in training the model,
    # but just in optimizing the parameters and extracting them as they directly represent the solution to our 
    # NMF problem.
    def fit(self):
        M_tensor = torch.from_numpy(self.M)
        optimizer = optim.Adam([self.A, self.W], lr=0.01)
        for iter in range(self.max_iter):

            optimizer.zero_grad()
            reconstruction = self.forward()
            loss = torch.norm(M_tensor - reconstruction, p='fro') # As indicated in the handout
            loss.backward()
            optimizer.step()
            
            # Clamping values in A and W to ensure they are non-negative
            self.A.data.clamp_(min=0)
            self.W.data.clamp_(min=0)
            
            print(f'{iter} loss: {loss.item()}')
            if loss.item() < self.tol:
                break
        print(f'Final loss: {loss.item()}')
        # Outputting factored matrices A and W into a file
        output_A = self.A.detach().numpy()
        output_W = self.W.detach().numpy()
        with open("final_A_SCMFDD_S_22", 'ab') as f: 
            np.savetxt(f, output_A)
        with open("final_W_SCMFDD_S_22", 'ab') as f: 
            np.savetxt(f, output_W)
        
# Initializing an instance of NMF and fitting it with our given input matrix
r= 22
np_matrix = np.array(matrix)
nmf = NMF(np_matrix, int(r))
nmf.fit()





