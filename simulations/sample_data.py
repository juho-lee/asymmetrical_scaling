import numpy as np
import torch

def sample_data(input_size, n, stdnoise):
    # sample observations on unit sphere
    vec = np.random.randn(input_size, n)
    vec /= np.linalg.norm(vec, axis=0)
    x = torch.from_numpy(vec.T.astype(np.float32))

    # x = torch.from_numpy(np.random.uniform(-1,1,(n,1))* np.random.dirichlet(np.ones((input_size, )), size=n))
    fun = lambda x: 5 * np.sin(np.pi * x)
    #ytrue = fun(x[:, 1]).reshape(
    #    -1, 1
    #)  # np.sin(2*np.pi*x.sum(axis=1)).reshape(-1, 1) # just a linear function
    ytrue = fun(x).mean(-1).reshape(-1, 1)# np.sin(2*np.pi*x.sum(axis=1)).reshape(-1, 1) # just a linear function
    
    # print(x.shape)
    # ytrue = fun(x[:,0]).reshape(-1, 1)
    epsilon = torch.from_numpy(
        stdnoise * np.random.normal(size=(n, 1)).astype(np.float32)
    )  # noise
    y = ytrue + epsilon
    return x, y, ytrue

input_size = 50
n = 100
ntest = 100
stdnoise = 0.1
x, y, ytrue = sample_data(input_size, n, stdnoise)
xtest, ytest, ytesttrue = sample_data(input_size, ntest, stdnoise)

torch.save(
    {
        "x": x,
        "y": y,
        "ytrue": ytrue,
        "xtest": xtest,
        "ytest": ytest,
        "ytesttrue": ytesttrue,
    },
    "data.th",
)
