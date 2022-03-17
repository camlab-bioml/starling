
import ...


def compute_v(Y, S, Theta):
    """
    compute the c x c tensor
    v^{d=1,\gamma=[c,c']}

    access parameters via
    mu = Theta[0] etc
    (see below)
    """
    ...

    return v

def compute_r(Y, S, Theta):
    """
    compute the c x 1 tensor
    r^{d=0,z=c}
    """
    ...

    return r

def Q(Theta, Y, S, r, v):
    """
    returns Q(\mathbf\Theta, \mathbf{Y}, \mathbf{s}, \mathbf{r}, \mathbf{v})
    """
    ...

    return q

def L(Theta, Y, S, r, v):
    """
    returns L(\mathbf\Theta, \mathbf{Y}, \mathbf{s}, \mathbf{r}, \mathbf{v})
    """
    ...


Theta = initialize_parameters(Y, S)

"""
Note: rather than deal with individual parameters constantly, we can make
Theta be a list of tensors
[mu, Sigma, psi, omega, lambda, pi, tau]
and always refer to them *within* the functions compute_v, compute_r, Q, L
when you need them, e.g.
mu = Theta[0]
etc. (this could also be a python dictionary!)
"""

opt = optim.Adam(Theta, lr=lr)

for i in range(N_ITER_EM):

    # E Step:
    with torch.no_grad():
        r = compute_r(Y, S, Theta)
        v = compute_v(Y, S, Theta)
    
    # M step (i.e. maximizing Q):
    for j in range(N_ITER_OPT):
        opt.zero_grad()
        q = Q(Theta, Y, S, r, v)
        opt.step()
    
    # Check for convergence
    l = L(Theta, Y, S, r, v)
