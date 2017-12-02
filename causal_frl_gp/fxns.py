import autograd
import autograd.numpy as np
import scipy, scipy.optimize
import pdb
from causal_frl_newest.causal_frl.distributions import truncated_normal
import itertools, copy
import python_utils.python_utils.basic as basic


ignore_c = False
#ignore_c = True
debug = False

def solve_system(A, b):
    ans, code = scipy.sparse.linalg.cg(A, b)
    if code < 0:
#        pdb.set_trace()
        assert False
    return ans

def fs_te_posterior(prec_ys, ys, xs, xs_te, k):
    N = len(xs)
    K = k(xs, xs)
    K_te = k(xs_te, xs)
    K_te = K_te.T
    K_te_te = k(xs_te, xs_te)
    L = np.linalg.cholesky(K + ((1./prec_ys)*np.eye(N)))
    mu_fs_te = np.dot(K_te, scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, ys, lower=True), lower=False))
    cov_fs_te = K_te_te - np.dot(K_te, scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, K_te, lower=True), lower=False))
    return mu_fs_te, cov_fs_te

def evidence(prec_ys, ys, K):
    N = len(K)
    L = np.linalg.cholesky(K + ((1./prec_ys)*np.eye(N)))
    return (
        (- 0.5 * np.dot(ys, scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, ys, lower=True), lower=False)))
        - np.sum(np.log(np.diag(L)))
        - ((N/2.) * np.log(2*np.pi))
        )

def truncated_normal_E_xxT(mu, prec):
    if ignore_c:
        return np.zeros((len(mu),len(mu)))
    else:
#    if len(mu) == 0:
#        return np.zeros((0,0))
#        return np.array([[]])
        return truncated_normal.E_x_xT((mu, prec))

def truncated_normal_E_x(mu, prec):
    if ignore_c:
        return np.zeros(mu.shape) # fix!
    else:
#    if len(mu) == 0:
#        return np.array([])
        try:
            ans = truncated_normal.mean((mu, prec))
            return ans
        except:
            pdb.set_trace()


def normal_E_x(mu, cov):
    return mu

def gamma_E_x(alpha, beta):
    return alpha / beta

def normal_E_xs_power(mus, vars, power):
    if power == 0:
        return np.ones(len(mus))
    elif power == 1:
        return mus
    elif power == 2:
        return (mus**2) + vars
    elif power == 3:
        return (mus**3) + (3*mus*vars)
    elif power == 4:
        return (mus**4) + (6*(mus**2)*vars) + (3*(vars**2))

def gamma_E_log_x(alpha, beta):
#    return np.log(alpha / beta) # fix
    return -np.log(beta) + scipy.special.psi(alpha)
    
def ls_to_zs(ls):
#    if ls.shape[1] == 0:
#        return np.zeros(shape=ls.shape)
#    pdb.set_trace()
    ts_bool = np.sum(ls, axis=1) > 0
    zs = np.zeros(ls.shape)
    zs[np.arange(len(ls)), np.argmax(ls, axis=1)] = 1
    zs[ts_bool,:] = 0
    return zs

def cs_to_zs(cs):
    zs = np.zeros(cs.shape)
    zs[np.arange(len(cs)), np.argmax(cs, axis=1)] = 1
    return zs

def zs_to_ls(zs):
    # zs is whether a sample matches each rule
    L = zs.shape[1]
    return (np.arange(0, L)[np.newaxis,:] >= np.argmax(zs, axis=1)[:,np.newaxis]).astype(int)

def E_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls):
    # new: should always return length N vector
    zs = ls_to_zs(ls)
#    pdb.set_trace()
    if isinstance(alphas_prec_ys, np.ndarray) and len(alphas_prec_ys.shape) > 0:
        E_prec_ys = np.dot(zs, gamma_E_x(alphas_prec_ys, betas_prec_ys))
    else:
        E_prec_ys = gamma_E_x(alphas_prec_ys, betas_prec_ys) * np.ones(len(ls))
#    print alphas_prec_ys, betas_prec_ys
    return E_prec_ys

def E_log_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls):
    # new: should always return length N vector
    zs = ls_to_zs(ls)
    if isinstance(alphas_prec_ys, np.ndarray) and len(alphas_prec_ys.shape) > 0:
        E_log_prec_ys = np.dot(zs, gamma_E_log_x(alphas_prec_ys, betas_prec_ys))
    else:
        E_log_prec_ys = gamma_E_log_x(alphas_prec_ys, betas_prec_ys) * np.ones(len(ls))
    return E_log_prec_ys

def expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0):
    E_prec_ys = E_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls)
    _expected_squared_error_ns = expected_squared_error_ns(mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power)
    E_log_prec_ys = E_log_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls)
    E_fs_0 = normal_E_xs_power(mus_fs, vars_fs, 0+fs_power)
    return expected_negloglik_ns_horse(E_fs_0, E_log_prec_ys, E_prec_ys, _expected_squared_error_ns, ls, ys)

def expected_negloglik_ns_horse(E_fs_0, E_log_prec_ys, E_prec_ys, expected_squared_error_ns, ls, ys):
#    ans = (E_prec_ys * expected_squared_error_ns)
#    ans2 = (-0.5 * (np.log(2 * np.pi) )) + (E_prec_ys * expected_squared_error_ns)
#    pdb.set_trace()
#    return ans2
#    print E_log_prec_ys
#    pdb.set_trace()
    #print E_prec_ys, 'precisions'
    #print E_log_prec_ys, 'log precisions'
#    return (0.5 * E_fs_0 * (np.log(2 * np.pi) + E_log_prec_ys)) + (E_prec_ys * expected_squared_error_ns) # fix: should first term be negative?
    return (0.5 * E_fs_0 * (np.log(2 * np.pi) - E_log_prec_ys)) + (E_prec_ys * expected_squared_error_ns) # fix: should first term be negative?
#    return (-0.5 * E_fs_0 * (np.log(2 * np.pi) - E_log_prec_ys)) + (E_prec_ys * expected_squared_error_ns)

def expected_squared_error_ns(mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0):
    E_c = truncated_normal_E_x(mu_c, prec_c)
    E_ccT = truncated_normal_E_xxT(mu_c, prec_c)
    E_fs_0 = normal_E_xs_power(mus_fs, vars_fs, 0+fs_power)
    E_fs = normal_E_xs_power(mus_fs, vars_fs, 1+fs_power)
    E_fs_squared = normal_E_xs_power(mus_fs, vars_fs, 2+fs_power)
    return expected_squared_error_ns_horse(E_c, E_ccT, E_fs, E_fs_squared, ls, ys, E_fs_0=E_fs_0)

def expected_squared_error_ns_horse(E_c, E_ccT, E_fs, E_fs_squared, ls, ys, E_fs_0=1.):
    # this is negative of what's in the exponent
    ans = - (
#        ((E_fs * (1. + np.dot(ls, E_c))) * ys)
        ((E_fs  + (E_fs_0 * np.dot(ls, E_c))) * ys) # ooh
        - (0.5 * (ys * ys * E_fs_0))
#        - (0.5 * (ys * ys * E_fs_squared))
#        - (0.5 * (E_fs_squared * E_fs_squared) + (np.dot(ls, E_c) * E_fs) + 0.5 * (np.sum(ls * np.dot(E_ccT,ls.T).T, axis=1)))
        - ((0.5 * E_fs_squared) + (np.dot(ls, E_c) * E_fs) + 0.5 * (np.sum(ls * np.dot(E_ccT,ls.T).T, axis=1) * E_fs_0)) # fix: check formula involving E_fs_0
        )
    #print np.sum(ans), 'squared error'
#    if E_fs_0 == 1. or E_fs == np.ones(len(ys)):
#        assert ans >= 0
#    if np.sum(ans) < 0:
#        print ((E_fs * (1. + np.dot(ls, E_c))) * ys)
#        print - (0.5 * (ys * ys * E_fs))
#        print - ((0.5 * E_fs_squared) + (np.dot(ls, E_c) * E_fs) + 0.5 * (np.sum(ls * np.dot(E_ccT,ls.T).T, axis=1) * E_fs_0))
#        pdb.set_trace()
    return ans

def lam_to_cov_fs_slow(K, lam):
#    return np.linalg.inv(np.linalg.inv(K) + np.diag(lam**1))
    return np.linalg.inv(np.linalg.inv(K) + np.diag(lam**2))

def v_bar_and_lam_bar(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=None, K_factors=None, cov_fs=None):
    # cov_fs passed in, as dF_dv_and_dF_dlam needs it, calls this, and already computed it
#    pdb.set_trace()
    if K_factors is None:
        mu_fs = np.dot(K, v)
    else:
        U, V = K_factors
        mu_fs = np.dot(U, np.dot(V, v))
    if cov_fs is None:
        assert False
        cov_fs = lam_to_cov_fs(K, lam) # fix
    mus_fs = mu_fs
    vars_fs = np.diag(cov_fs)

    pow_0 = expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0)
    pow_1 = expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=1)
    pow_2 = expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=2)

    v_bar = ((mus_fs * pow_0) - pow_1) / vars_fs
    lam_bar = ((pow_2 - (2 * mus_fs * pow_1) + ((mus_fs**2) * pow_0)) - (vars_fs * pow_0)) / (vars_fs**2)

    return v_bar, lam_bar

def negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, fs, ls, ys):
    # no expectation over fs
    E_c = truncated_normal_E_x(mu_c, prec_c)
    E_ccT = truncated_normal_E_xxT(mu_c, prec_c)
    E_prec_ys = E_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls)
    squared_error_ns = expected_squared_error_ns_horse(E_c, E_ccT, fs, fs * fs, ls, ys)
    E_log_prec_ys = E_log_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls)
    fs_power = 0
    E_fs_0 = 1.
    return expected_negloglik_ns_horse(E_fs_0, E_log_prec_ys, E_prec_ys, squared_error_ns, ls, ys)
#    return (0.5 * E_fs_0 * ((np.log(2 * np.pi) + E_log_prec_ys))) + (E_prec_ys * expected_squared_error_ns)
#    return E_prec_ys * expected_squared_error_ns_horse(E_c, E_ccT, fs, fs * fs, ls, ys)

def sampling_expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0, num_samples=5000):
    N = len(ys)
    samples = np.zeros((num_samples,N))
    for i in xrange(num_samples):
        fs_sample = np.random.normal(mus_fs, vars_fs**0.5)
        samples[i,:] = (fs_sample**fs_power) * negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, fs_sample, ls, ys)
    return np.mean(samples, axis=0)

def sampling_v_bar_and_lam_bar(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, cov_fs=None, num_samples=5000, debug=False):
    # cov_fs passed in, as dF_dv_and_dF_dlam needs it, calls this, and already computed it

    mu_fs = np.dot(K, v)
    if cov_fs is None:
        cov_fs = lam_to_cov_fs(K, lam) # fix
    mus_fs = mu_fs
    vars_fs = np.diag(cov_fs)

    if debug:
        print 'mu_fs'
        print mu_fs
        print 'vars_fs'
        print vars_fs

    def v_bar_val(fs):
        return (mu_fs - fs) * negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, fs, ls, ys) / vars_fs
    
    def lam_bar_val(fs):
        negloglik_ns_val = negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, fs, ls, ys)
        return (((fs - mu_fs) ** 2) * negloglik_ns_val / (vars_fs ** 2)) - (negloglik_ns_val / vars_fs)

    v_bar = np.mean([v_bar_val(np.random.normal(mu_fs, vars_fs**0.5)) for i in xrange(num_samples)], axis=0)
    lam_bar = np.mean([lam_bar_val(np.random.normal(mu_fs, vars_fs**0.5)) for i in xrange(num_samples)], axis=0)

    return v_bar, lam_bar

def dF_fs_dv_and_dlam(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K):

#    cov_fs = lam_to_cov_fs(K, lam) # fix

#    v_bar, lam_bar = v_bar_and_lam_bar(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, cov_fs)
    v_bar, lam_bar = v_bar_and_lam_bar(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam**2, ls, ys, K=K, K_factors=K_factors, cov_fs=cov_fs)

    dF_fs_dv = np.dot(K, (v - v_bar))
#    dF_fs_dlam = 0.5 * np.dot(cov_fs * cov_fs, lam - lam_bar)
    dF_fs_dlam = 0.5 * np.dot(cov_fs * cov_fs, lam**2 - lam_bar)
    actual_dF_fs_dlam = 2 * lam * dF_fs_dlam # fix?
#    return dF_fs_dv, dF_fs_dlam
    return dF_fs_dv, actual_dF_fs_dlam

def F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=None, K_factors=None, K_inv=None, K_inv_factors=None, cov_fs=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False): # fix: add trace_method to get_variational_fs
    # only contains terms that depend on q_fs

    mu_fs = np.dot(K, v)
    K_inv = np.linalg.inv(K)
    if cov_fs is None:
        cov_fs = lam_to_cov_fs(K, lam) # fix
    mus_fs = mu_fs
    vars_fs = np.diag(cov_fs)
    N = len(ys)

    if debug:
        print np.sum(expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys)), 'fs 1'
        print (0.5 * np.sum(K_inv * cov_fs)), 'fs 2'
        print (0.5 * np.dot(v.T, np.dot(K, v))), 'fs 3'
        print - (0.5 * np.log(np.linalg.det(cov_fs))), 'fs 4'

    if trace_method == 'direct':
        if K_inv_factors is None:
            K_inv_cov_fs_trace = np.sum(K_inv * cov_fs)
        else:
            U, V = K_inv_factors
            R = np.dot(V, cov_fs)
            L = U
            K_inv_cov_fs_trace = np.sum(L * R.T)
    else:
        assert not (K_factors is None)
        K_inv_cov_fs_trace = np.sum(K_inv_cov_fs_diag_low_rank(K_factors, lam, inv_method))

    if K_factors is None:
        vKv = np.dot(v.T, np.dot(K, v))
    else:
        U, V = K_factors
        vKv = np.dot(np.dot(v.T, U), np.dot(V, v))

    if K_inv_factors is None:
        if log_det_method == 'vanilla':
            log_det_cov_fs = np.log(np.linalg.det(cov_fs))
    else:
        log_det_cov_fs = K_inv_cov_fs_diag_low_rank(K_factors, lam, log_det_method)
    
    return (
        + np.sum(expected_negloglik_ns(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, mus_fs, vars_fs, ls, ys))
#        + np.sum(K_inv * cov_fs) # fix, possibly for efficiency
        + (0.5 * K_inv_cov_fs_trace)
        + (0.5 * vKv)
        - (0.5 * log_det_cov_fs) # fix: should compute determinant using cholesky factor
        - ((N/2.) * np.log(2 * np.pi * np.e))
        + ((N/2.) * np.log(2 * np.pi)) # part of log Z_0
        + (0.5 * np.log(np.linalg.det(K))) # part of log Z_0.  fix: should precompute?
        )

def dF_fs_dtheta(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, dK_dtheta):
    cov_fs = lam_to_cov_fs(K, lam)
    v_bar, lam_bar = v_bar_and_lam_bar(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, cov_fs)
#    print v_bar, lam_bar
    B_bar = K + np.diag(1. / lam_bar)
    return - 0.5 * np.sum((np.outer(v_bar, v_bar) - np.linalg.inv(B_bar))[:,:,np.newaxis] * dK_dtheta, axis=(0,1)) # assumes dK_dtheta has theta in 3rd dimension, not 1st

def variational_fs_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=None, K_factors=None, K_inv=None, K_inv_factors=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False):
    # define versions of F and its gradient that accept scalars
    N = len(ys)
    assert len(v) == len(lam) == N

    if debug:
        old_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)
    
    def scalar_F_and_dF_ds(s):
        v = s[0:N]
        lam = s[N:]
        cov_fs = lam_to_cov_fs(K, lam, K_inv_factors, inv_method=inv_method) # fix: need to make new wrapper for lam_to_cov_fs
        val = F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=K, K_factors=K_factors, K_inv=K_inv, K_inv_factors=K_inv_factors, cov_fs=cov_fs, inv_method=inv_method, log_det_method=log_det_method, trace_method=trace_method, debug=debug) # fix: redundant calculations, like calculating cov_fs, expected values of stuff
        dF_dv, dF_dlam = dF_fs_dv_and_dlam(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=K, K_factors=K_factors, cov_fs=cov_fs)
        dF_ds = np.concatenate((dF_dv, dF_dlam))
        return val, dF_ds

    if debug:
        assert scipy.optimize.check_grad(lambda v: F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K), lambda v: dF_fs_dv_and_dlam(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)[0], v) < 0.01
        assert scipy.optimize.check_grad(lambda lam: F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K), lambda v: dF_fs_dv_and_dlam(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)[1], lam) < 0.01
    
    s = np.concatenate((v, lam))
    new_s = scipy.optimize.minimize(scalar_F_and_dF_ds, s, jac=True)['x']
    new_v, new_lam = new_s[0:N], new_s[N:]

    if debug:
        new_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, new_v, new_lam, ls, ys, K)
        print 'q_fs', 'new_F:', new_F, 'old_F', old_F
        assert new_F <= old_F
    
    return new_v, new_lam

def F_c(mu_c_0, prec_c_0, mu_c, prec_c):
    E_c = truncated_normal_E_x(mu_c, prec_c)
    E_ccT = truncated_normal_E_xxT(mu_c, prec_c)
#    pdb.set_trace()
    if debug:
        print - np.dot(np.dot(prec_c_0, mu_c_0), E_c), 'c 1'
        print (0.5 * np.sum(prec_c_0 * E_ccT)), 'c 2'
        print (0.5 * np.log(np.linalg.det(prec_c))), 'c 3'
    ans = - sum((
        np.dot(np.dot(prec_c_0, mu_c_0), E_c),
        - (0.5 * np.sum(prec_c_0 * E_ccT)),
#        + (0.5 * np.log(np.linalg.det(prec_c))) # negative entropy, or fix: should be positive entropy?
        - (0.5 * np.log(np.linalg.det(prec_c))) # negative entropy, or fix: should be positive entropy?
#        + (0.5 * np.log(np.linalg.det(prec_c_0))) # negative entropy
        ))
    return ans

def variational_c_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, debug=False):

    if debug:
        old_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)
    
#    E_prec_ls = gamma_E_x(alphas_prec_ys, betas_prec_ys)
    E_prec_ys = E_prec_ys_helper(alphas_prec_ys, betas_prec_ys, ls)
    new_prec_c = prec_c_0 + np.dot(E_prec_ys[np.newaxis,:] * ls.T, ls)
#    if isinstance(E_prec_ls, np.ndarray) and len(alphas_prec_ys.shape) > 0: # new
#        new_prec_c = prec_c_0 + (E_prec_ls[:,np.newaxis] * np.dot(ls.T, ls))
#        new_prec_c = prec_c_0 + (E_prec_ls * np.dot(E_prec_ys[np.newaxis,:] * ls.T, ls))
#    else:
#        new_prec_c = prec_c_0 + (E_prec_ls * np.dot(ls.T, ls))
    try:
        assert (new_prec_c == new_prec_c.T).all()
    except:
        pdb.set_trace()
    zs = ls_to_zs(ls)
    mus_fs = np.dot(K, v) # fix: expensive
    cov_fs = lam_to_cov_fs(K, lam) # fix: expensive
    vars_fs = np.diag(cov_fs)
    
    E_fs = normal_E_xs_power(mus_fs, vars_fs, 1)

    new_prec_mu_c = np.dot(prec_c_0, mu_c_0) + np.dot(ls.T, E_prec_ys * (ys - E_fs))
#    if isinstance(E_prec_ls, np.ndarray) and len(alphas_prec_ys.shape) > 0:
#        new_prec_mu_c = np.dot(prec_c_0, mu_c_0) + np.dot(ls.T, np.dot(zs, E_prec_ls) * (ys - E_fs))
#    else:
#        new_prec_mu_c = np.dot(prec_c_0, mu_c_0) + np.dot(ls.T, E_prec_ls * (ys - E_fs))
    if new_prec_mu_c.shape[0] > 0:
        new_mu_c = solve_system(new_prec_c, new_prec_mu_c)
    else:
        new_mu_c = np.zeros((0,))

    if debug:
        new_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, new_mu_c, new_prec_c, v, lam, ls, ys, K)
        print 'q_c', 'new_F:', new_F, 'old_F', old_F
        if new_F > old_F:

            print 'q_c does not decrease, increase:', new_F - old_F
#        assert new_F <= old_F


    return new_mu_c, new_prec_c

def F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K):
    return (
        + F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)
        + F_c(mu_c_0, prec_c_0, mu_c, prec_c)
        + F_prec_ys(alphas_prec_ys_0, betas_prec_ys_0, alphas_prec_ys, betas_prec_ys)
        )

def F_prec_ys(alphas_prec_ys_0, betas_prec_ys_0, alphas_prec_ys, betas_prec_ys):
    E_log_prec_ys = gamma_E_log_x(alphas_prec_ys, betas_prec_ys) # fix: should handle scalar and array alphas_prec_ys
    E_prec_ys = gamma_E_x(alphas_prec_ys, betas_prec_ys) # these are actually either of length L, or scalars, not of length N
#    entropy = (alphas_prec_ys - np.log(betas_prec_ys) + scipy.special.gammaln(alphas_prec_ys) + (1.-alphas_prec_ys) * scipy.special.psi(alphas_prec_ys)) # negative entropy. fix: should be positive entropy?
#    print entropy, 'prec_ys entropy'
    if debug:
        print - np.sum(((alphas_prec_ys_0 - 1) * E_log_prec_ys)), 'prec_ys 1'
        print np.sum((betas_prec_ys_0 * E_prec_ys)), 'prec_ys 2'
        print  (alphas_prec_ys - np.log(betas_prec_ys) + scipy.special.gammaln(alphas_prec_ys) + (1.-alphas_prec_ys) * scipy.special.psi(alphas_prec_ys)), 'prec_ys 3'
#    pdb.set_trace()
    ans = - np.sum((
        ((alphas_prec_ys_0 - 1) * E_log_prec_ys),
        - (betas_prec_ys_0 * E_prec_ys),
#        - (alphas_prec_ys - np.log(betas_prec_ys) + scipy.special.gammaln(alphas_prec_ys) + (1.-alphas_prec_ys) * scipy.special.psi(alphas_prec_ys)) # negative entropy
        + (alphas_prec_ys - np.log(betas_prec_ys) + scipy.special.gammaln(alphas_prec_ys) + (1.-alphas_prec_ys) * scipy.special.psi(alphas_prec_ys)) # negative entropy. fix: should be positive entropy?
        ))
    return ans

def variational_prec_ys_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, debug=False):
    
    if debug:
        old_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)

    zs = ls_to_zs(ls)
    mus_fs = np.dot(K, v) # fix: expensive
    cov_fs = lam_to_cov_fs(K, lam) # fix: expensive
    vars_fs = np.diag(cov_fs)
#    pdb.set_trace()
    if isinstance(alphas_prec_ys, np.ndarray) and len(alphas_prec_ys.shape) > 0:
        new_alphas_prec_ys = alphas_prec_ys_0 + (0.5 * np.sum(zs, axis=0))
        new_betas_prec_ys = betas_prec_ys_0 + np.dot(zs.T, expected_squared_error_ns(mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0)) # fix: check if factor of 2 off
    else:
        new_alphas_prec_ys = alphas_prec_ys_0 + (0.5 * np.sum(zs))
        new_betas_prec_ys = betas_prec_ys_0 + np.sum(expected_squared_error_ns(mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0)) # fix: check if factor of 2 off
#    print np.sum(expected_squared_error_ns(mu_c, prec_c, mus_fs, vars_fs, ls, ys, fs_power=0)) , 'asdf'
#    pdb.set_trace()
    if debug:
        new_F = F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, new_alphas_prec_ys, new_betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K)
        print 'q_prec_ys', 'new_F:', new_F, 'old_F', old_F
        assert new_F <= old_F
#    print new_alphas_prec_ys, new_betas_prec_ys, np.sum(zs)
    return new_alphas_prec_ys, new_betas_prec_ys

def posterior_diff(posterior_1, posterior_2):
    alphas_prec_ys_1, betas_prec_ys_1, mu_c_1, prec_c_1, v_1, lam_1 = posterior_1
    alphas_prec_ys_2, betas_prec_ys_2, mu_c_2, prec_c_2, v_2, lam_2 = posterior_2
    return np.linalg.norm(v_1 - v_2) + np.linalg.norm(lam_1 - lam_2) + np.linalg.norm(mu_c_1 - mu_c_2) + np.linalg.norm(prec_c_1 - prec_c_2) + np.linalg.norm(alphas_prec_ys_1 - alphas_prec_ys_2) + np.linalg.norm(betas_prec_ys_1 - betas_prec_ys_2)

def variational_posterior(num_tries, max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, K=None, K_factors=None, K_inv=None, K_inv_factors=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False, eps=0.001):

    L = ls.shape[1]
    N = len(ys)

    evidences = []
    posteriors = []

    for i in xrange(num_tries):
    
        # initialize variational params
        state = np.random.get_state()

        np.random.seed(i)
        alphas_prec_ys = np.random.uniform(0.5, 2.0, size=np.shape(alphas_prec_ys_0)) # fix
        betas_prec_ys = np.random.uniform(0.5, 2.0, size=np.shape(alphas_prec_ys_0))
        mu_c = np.random.uniform(0., 1., size=L)
        prec_c = np.diag(np.random.uniform(0.5, 2.0, size=L))
        v = np.random.normal(size=N)
        lam = np.random.uniform(0.5, 2., size=N)

        np.random.set_state(state)

        old_posterior = None
    
        # cycle through update steps until convergence
        for j in xrange(max_iter):

            if debug >= 2: print 'variational step:', j
            alphas_prec_ys, betas_prec_ys = variational_prec_ys_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, debug=debug)
            mu_c, prec_c = variational_c_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, debug=debug)
            v, lam = variational_fs_posterior(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K=K, K_factors=K_factors, K_inv=K_inv, K_inv_factors=K_inv_factors, inv_method=inv_method, log_det_method=log_det_method, trace_method=trace_method, debug=debug)

            new_posterior = (alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam)
            if not (old_posterior is None):
                diff = posterior_diff(new_posterior, old_posterior)
                if debug: print 'diff', diff
                if diff < eps:
                    break
            old_posterior = new_posterior
            
        evidences.append(F(alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K))
        posteriors.append(new_posterior)

    if debug:
        print 'evidences over tries:', evidences

    best_i = np.argmin(evidences)
    return posteriors[best_i], evidences[best_i]
#    return alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam

def optimize_theta(theta_method, theta_num_tries, theta_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, xs, k, theta_init_f,theta_grid=None, k_factors=None, k_inv = None, k_inv_factors=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False, eps=0.001):
    # gradient free hyperparam optimization.  returns best theta and associated evidence

    def objective(theta):
        if k_factors is None:
            K = k(theta, xs, xs)
            K_factors = None
        else:
            K_factors = k_factors(theta, xs, xs)
            K = None
        if k_inv_factors is None:
            K_inv = k_inv(theta, xs, xs)
            K_inv_factors = None
        else:
            K_inv_factors = k_inv_factors(theta, xs, xs)
            K_inv = None
        posterior, evidence = variational_posterior(variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, K, K_factors=None, K_inv=None, K_inv_factors=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False, eps=0.001)

        return evidence
    
    if theta_method != 'grid':

        thetas = []
        evidences = []
        
        for i in xrange(theta_num_tries):

            state = np.random.get_state()
            np.random.seed(i)
            theta_init = theta_init_f()
            np.random.set_state(state)
            result = scipy.optimize.minimize(objective, theta_init, method=theta_method, options={'maxiter': theta_max_iter, 'disp':bool(debug)})

            theta = result['x']
            evidence = result['fun']
            thetas.append(theta)
            evidences.append(evidence)

        if debug: print 'thetas and evidences:', zip(thetas, evidences)
            
        best_i = np.argmin(evidences)
        return thetas[best_i], evidences[best_i]

    else:
        import itertools
        d = {}
        for (i,theta) in enumerate(itertools.product(*theta_grid)):
            if debug: print 'iter:', i, 'theta:', theta
            d[theta] = objective(theta)
        sorted_tuples = sorted(d.iteritems(), key=lambda (theta,evidence): evidence)
        if debug: print 'thetas and evidences:', sorted_tuples
        best_theta, best_evidence = sorted_tuples[0]
        return best_theta, best_evidence

def rule_search(search_num_tries, search_num_iter, theta_method, theta_num_tries, theta_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, all_zs, ts, ys, xs, k=None, theta_init_f=None, rule_idxs_init_f=None, theta_grid=None, k_factors=None, k_inv = None, k_inv_factors=None, inv_method='vanilla', log_det_method='vanilla', trace_method='direct', debug=False, eps=0.001, min_support=2):
    # moves are insert, remove, permute, replace

    def objective(rule_idxs):
        zs = np.concatenate((all_zs[:,rule_idxs], np.ones((len(ys),1))), axis=1)
        ls = zs_to_ls(zs) * ts[:,np.newaxis]
        theta, evidence = optimize_theta(theta_method, theta_num_tries, theta_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, xs, k, theta_init_f, debug=debug, theta_grid=theta_grid, k_factors=k_factors, k_inv=k_inv, k_inv_factors=k_inv_factors, inv_method=inv_method, log_det_method=log_det_method, trace_method=trace_method, debug=debug, eps=eps)
        return evidence

    def pre_reject(rule_idxs):
        zs = np.concatenate((all_zs[:,rule_idxs], np.ones((len(ys),1))), axis=1)
        treated_ls = np.sum(zs * ts, axis=0)
        control_ls = np.sum(zs * (ts==0).astype(int), axis=0)

        if debug: print 'treated_ls:', treated_ls, 'control_ls', control_ls
        if np.min(treated_ls) < min_support or np.min(control_ls) < min_support:
            return True
        return False
    
    rule_idxss = []
    objectives = []


    num_rules = all_cs.shape[1]
    all_rule_idxs = np.arange(num_rules, dtype=int)
    
    for i in xrange(search_num_tries):

        state = np.random.get_state()
        np.random.seed(i)
        rule_idxs = rule_idxs_init_f()
        np.random.set_state(state)
        old_objective = objective(rule_idxs)
            
        for j in xrange(search_num_iter):


#            np.random.seed(j)

            old_rule_idxs = copy.deepcopy(rule_idxs)

            possible_moves = []
            if len(rule_idxs) < all_cs.shape[1]:
                possible_moves.append('insert')
            if len(rule_idxs) > 0:
                possible_moves.append('delete')
                possible_moves.append('replace')
            if len(rule_idxs) >= 2:
                possible_moves.append('swap')

            which_move = np.random.choice(possible_moves)
            
            if which_move == 'insert':

                insert_rule_idx = None
                while (insert_rule_idx is None) or (insert_rule_idx in rule_idxs):
                    insert_rule_idx = np.random.choice(all_rule_idxs)
#                    insert_rule_idx = np.random.choice([i for i in all_rule_idxs if i in rule_idxs])

                insert_pos = np.random.randint(0, len(rule_idxs)+1)
                rule_idxs = np.insert(rule_idxs, insert_pos, insert_rule_idx)

            if which_move == 'delete':
                delete_pos = np.random.randint(0, len(rule_idxs))
                rule_idxs = np.delete(rule_idxs, delete_pos)

            if which_move == 'replace':

                replace_rule_idx = None
                while (replace_rule_idx is None) or (replace_rule_idx in rule_idxs):
                    replace_rule_idx = np.random.choice(all_rule_idxs)
                replace_pos = np.random.randint(0, len(rule_idxs))
                rule_idxs[replace_pos] = replace_rule_idx

            if which_move == 'swap':
                pos1, pos2 = np.random.choice(rule_idxs, 2)
                temp = rule_idxs[pos1]
                rule_idxs[pos1] = rule_idxs[pos2]
                rule_idxs[pos2] = temp


            if debug: print 'move:', old_rule_idxs, which_move, rule_idxs


            if pre_reject(rule_idxs):
                if debug: print 'pre_rejected'
                accept = False
            else:
                new_objective = objective(rule_idxs)
                if debug: print 'new_objective:', new_objective, 'old_objective:', old_objective
                accept = new_objective < old_objective
            if accept:
                old_objective = new_objective
            else:
                rule_idxs = old_rule_idxs

        rule_idxss.append(rule_idxs)
        objectives.append(old_objective)

    best_i = np.argmin(objectives)
    return rule_idxss[best_i], objectives[best_i]


def gaussian_K(theta, xs1, xs2):

    sigma, mult = theta
    
    diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
    norms = np.sum(diff * diff, axis=2)

    K = (mult**2) * np.exp(-1. * norms / (2 * (sigma**2)))

    return K

dgaussian_K_dsigma = autograd.jacobian(gaussian_K, argnum=0)

def split(proportion, all_cs, ts, ys, xs, es):
    N_tr = int(len(ts) * proportion)
    all_cs_tr, all_cs_te = all_cs[0:N_tr], all_cs[N_tr:]
    ts_tr, ts_te = ts[0:N_tr], ts[N_tr:]
    ys_tr, ys_te = ys[0:N_tr], ys[N_tr:]
    xs_tr, xs_te = xs[0:N_tr], xs[N_tr:]
    if not (es is None):
        es_tr, es_te = es[0:N_tr], es[N_tr:]
    else:
        es_tr, es_te = None, None
    return (all_cs_tr, ts_tr, ys_tr, xs_tr, es_tr), (all_cs_te, ts_te, ys_te, xs_te, es_te)

class cfrl_predictor(object):

    def __init__(self, rule_idxs, mu_c, prec_c):
        self.rule_idxs, self.mu_c, self.prec_c = rule_idxs, mu_c, prec_c

    def predict(self, all_cs, xs):
        means = truncated_normal_E_x(self.mu_c, self.prec_c)
        return np.dot(zs_to_ls(cs_to_zs(np.concatenate((all_cs[:,self.rule_idxs], np.ones((len(xs),1))), axis=1))), means)

    def get_rule_idxs_hat(self):
        return self.rule_idxs

class cfrl_fitter(object):

    def __init__(self, search_num_tries, search_num_iter, theta_method, theta_num_tries, theta_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0_f, betas_prec_ys_0_f, mu_c_0_f, prec_c_0_f, k, theta_init_f, rule_idxs_init_f, debug=False, theta_grid=None, min_support=2):
        self.search_num_tries, self.search_num_iter, self.theta_method, self.theta_num_tries, self.theta_max_iter, self.variational_num_tries, self.variational_max_iter, self.alphas_prec_ys_0_f, self.betas_prec_ys_0_f, self.mu_c_0_f, self.prec_c_0_f, self.k, self.theta_init_f, self.rule_idxs_init_f, self.debug, self.theta_grid, self.min_support = search_num_tries, search_num_iter, theta_method, theta_num_tries, theta_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0_f, betas_prec_ys_0_f, mu_c_0_f, prec_c_0_f, k, theta_init_f, rule_idxs_init_f, debug, theta_grid, min_support
        

    def fit(self, all_cs, ts, ys, xs, rule_idxs=None):
        rule_idxs_fit, objective = rule_search(self.search_num_tries, self.search_num_iter, self.theta_method, self.theta_num_tries, self.theta_max_iter, self.variational_num_tries, self.variational_max_iter, self.alphas_prec_ys_0_f, self.betas_prec_ys_0_f, self.mu_c_0_f, self.prec_c_0_f, all_cs, ts, ys, xs, self.k, self.theta_init_f, self.rule_idxs_init_f, self.debug, self.theta_grid, self.min_support)
        alphas_prec_ys_0 = self.alphas_prec_ys_0_f(rule_idxs_fit)
        betas_prec_ys_0 = self.betas_prec_ys_0_f(rule_idxs_fit)
        mu_c_0 = self.mu_c_0_f(rule_idxs_fit)
        prec_c_0 = self.prec_c_0_f(rule_idxs_fit)
        ls = zs_to_ls(cs_to_zs(np.concatenate((all_cs[:,rule_idxs_fit], np.ones((len(ys),1))), axis=1))) * ts[:,np.newaxis]
        theta_fit, evidence1 = optimize_theta(self.theta_method, self.theta_num_tries, self.theta_max_iter, self.variational_num_tries, self.variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, xs, self.k, self.theta_init_f, self.debug, self.theta_grid)
        K = self.k(theta_fit, xs, xs)
        posterior, evidence2 = variational_posterior(self.variational_num_tries, self.variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, K, debug=self.debug, eps=0.001)
        assert evidence1 == evidence2
        (alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam) = posterior
        return cfrl_predictor(rule_idxs_fit, mu_c, prec_c)

class basic_data_getter(object):

    def __init__(self, data_dim, p_treat, p_match, num_rules, f, ds_getter, L, noise_sd):
        self.data_dim, self.p_treat, self.p_match, self.num_rules, self.f, self.ds_getter, self.L, self.noise_sd = data_dim, p_treat, p_match, num_rules, f, ds_getter, L, noise_sd

    def __call__(self, i, N):
        state = np.random.get_state()
        np.random.seed(i)
        xs = np.random.normal(size=(N,self.data_dim))
        ds = self.ds_getter(self.L)
        all_cs = (np.random.uniform(size=(N,self.num_rules)) < self.p_match).astype(int)
        ts = (np.random.uniform(size=N) < self.p_treat).astype(int)
        rule_idxs = np.arange(self.L-1)
        cs = np.concatenate((all_cs[:,rule_idxs], np.ones((N,1))), axis=1)
        ls = fxns.zs_to_ls(fxns.cs_to_zs(cs))
        es = np.dot(ls, ds)
        fs = f(i, xs)
        noises = np.random.normal(scale=self.noise_sd, size=N)
        ys = (es * ts) + fs + noises
        np.random.set_state(state)
        return all_cs, ts, ys, xs, es, rule_idxs, ds

"""
"""


def old_optimize_theta(grad_num_tries, grad_max_iter, variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, xs, k, k_grad, theta_init_f):
    # k's arguments are hyperparameter vector, xs1, xs2, returns kernel matrix
    # k_grad's arguments are the same
    # should give optional starting q to variational posterior if not the first term called in maximize_evidence's grad descent, in which case variational max_iter is ignored

    def F_and_dF_dtheta(theta):
        K = k(theta, xs, xs)
        (alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam), evidence = variational_posterior(variational_num_tries, variational_max_iter, alphas_prec_ys_0, betas_prec_ys_0, mu_c_0, prec_c_0, ls, ys, K)
#        val = F_fs(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K) + F_c(mu_c_0, prec_c_0, mu_c, prec_c) + F_prec_ys(alphas_prec_ys_0, betas_prec_ys_0, alphas_prec_ys, betas_prec_ys) # duplicates variational_evidence
        dK_dtheta = k_grad(theta, xs, xs)
        grad = dF_fs_dtheta(alphas_prec_ys, betas_prec_ys, mu_c, prec_c, v, lam, ls, ys, K, dK_dtheta)
        if debug:
            print 'evidence:', evidence, 'grad:', grad, 'theta:', theta
#        pdb.set_trace()
        return evidence, grad

    theta_init = theta_init_f()
    return scipy.optimize.minimize(F_and_dF_dtheta, theta_init, jac=True)['x']


def gaussian_K(theta, xs1, xs2):

    sigma, mult = theta
    
    diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
    norms = np.sum(diff * diff, axis=2)

    K = (mult**2) * np.exp(-1. * norms / (2 * (sigma**2)))

    return K

dgaussian_K_dsigma = autograd.jacobian(gaussian_K, argnum=0)

def SOR_gaussian_K(theta, xs1, xs2, idxs=np.arange(10,dtype=int)):
    subset = xs1[idxs,:]
    Q = gaussian_K(theta, xs1, subset)
    M = np.linalg.inv(gaussian_K(theta, subset, subset))
    return Q, M, Q.T

def factor_K_inv(K):
    pass

def lam_to_cov_fs(K, lam, K_inv_factors=None, inv_method='vanilla'):
    if K_inv_factors is None:
        return lam_to_cov_fs_slow(K, lam)
    else:
        return lam_to_cov_fs_low_rank(K_inv_factors, lam, inv_method=inv_method, just_var_fs=False)

def lam_to_cov_fs_low_rank(K_inv_factors, lam, inv_method='vanilla', just_var_fs=False):
    # gives diagonal of cov_fs
    lam_vec = lam**2
    K_inv_U, K_inv_V = K_inv_factors
    M = K_inv_U.shape[1]
    lam_vec_inv = 1. / lam_vec
    U, V = K_inv_U, K_inv_V
    S = np.eye(M) + np.dot(V, lam_vec_inc[:,np.newaxis] * U)
    if inv_method == 'vanilla':
        S_inv = np.linalg.inv(S)
    else:
        assert False
    R = np.dot(S_inv, V) * lam_vec_inv[np.newaxis,:]
    L = lam_vec_inv[:,np.newaxis] * U
    if just_var_fs:
        var_fs = lam_vec_inv - np.sum(L * R.T, axis=1)
        return var_fs
    else:
        cov_fs = np.diag(lam_vec_inv) - np.dot(L, R)
        return cov_fs

def factor_K(K):
    pass

def K_inv_cov_fs_diag_low_rank(K_factors, lam, inv_method='vanilla'):
    K_U, K_V = K_factors
    U, V = (lam**2)[:,np.newaxis] * K_U, K_V
    N, M = K_U.shape
    S = np.eye(M) + np.dot(V,U)
    if inv_method == 'vanilla':
        S_inv = np.linalg.inv(S)
    else:
        assert False
    R = np.dot(S_inv, V)
    L = U
    ans = np.ones(N) - np.sum(L * R.T, axis=1)
    return ans

def log_det_cov_fs_low_rank(K_inv_factors, lam, log_det_method='vanilla'):
    K_inv_U, K_inv_V = K_inv_factors
    lam_vec = lam**2
    lam_vec_inv = 1. / (lam_vec)
    M = U.shape[1]
    S = np.eye(M) + np.dot(V, lam_vec_inv[:,np.newaxis] * U)
    if det_method == 'vanilla':
        log_det_S = np.log(np.linalg.det(S))
    else:
        assert False
    return -(log_det_S + np.sum(np.log(lam_vec)))

