# https://github.com/eyurtsev/FlowCytometryTools/issues/44
import collections
from collections import abc
from numbers import Number
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy.external as sce
import torch

collections.MutableMapping = abc.MutableMapping
from flowsom import flowsom
from scanpy import AnnData
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConcatDataset(Dataset):
    """A dataset composed of datasets

    :param datasets: the datasets to concatenate, each of ``d.shape[0] == m``
    """

    def __init__(self, datasets: list[torch.Tensor]):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def init_clustering(
    initial_clustering_method: Literal["User", "KM", "GMM", "FS", "PG"],
    adata: AnnData,
    k: Union[int, None] = None,
    labels: Optional[np.ndarray] = None,
) -> AnnData:
    """Compute initial cluster centroids, variances & labels

    :param adata: The initial data to be analyzed
    :param initial_clustering_method: The method for computing the initial clusters,
        one of ``KM`` (KMeans), ``GMM`` (Gaussian Mixture Model),
        ``FS`` (FlowSOM), ``User`` (user-provided), or ``PG`` (PhenoGraph).
    :param k: The number of clusters, must be ``n_components`` when ``initial_clustering_method`` is ``GMM`` (required),
        ``k`` when ``initial_clustering_method`` is ``KM`` (required), ``k`` when ``initial_clustering_method``
        is ``FS`` (required), ``?`` when  ``initial_clustering_method`` is ``PG`` (optional), and can be ommited when
        ``initial_clustering_method`` is "User", because user will be passing in their own labels.
    :param labels: optional, user-provided labels

    :raises: ValueError

    :return: The annotated data with labels, centroids, and variances
    """

    if initial_clustering_method not in ["KM", "GMM", "FS", "PG", "User"]:
        raise ValueError(
            'initial_clustering_method must be one of "KM","GMM","FS","PG" or "User" defined cluster centroids/variances'
        )

    if initial_clustering_method in ["KM", "GMM", "FS"] and k is None:
        raise ValueError(
            "k cannot be ommitted for KMeans, FlowSOM, or Gaussian Mixture"
        )

    if initial_clustering_method == "User" and labels is None:
        raise ValueError(
            "labels must be provided when initial_clustering_method is set to 'User'"
        )

    if initial_clustering_method == "KM":
        kms = KMeans(k).fit(adata.X)
        init_l = kms.labels_
        init_label_class = np.unique(init_l)

        init_e = kms.cluster_centers_
        init_ev = np.array(
            [np.array(adata.X)[init_l == c, :].var(0) for c in init_label_class]
        )

    elif initial_clustering_method == "GMM":
        gmm = GaussianMixture(n_components=k, covariance_type="diag").fit(adata.X)
        init_l = gmm.predict(adata.X)

        init_e = gmm.means_
        init_ev = gmm.covariances_

    elif initial_clustering_method == "User" or initial_clustering_method == "PG":
        if initial_clustering_method == "PG":
            init_l, _, _ = sce.tl.phenograph(adata.X)
        else:
            init_l = labels

        classes = np.unique(init_l)
        k = len(classes)
        init_e = np.zeros((k, adata.X.shape[1]))
        init_ev = np.zeros((k, adata.X.shape[1]))
        for i, c in enumerate(classes):
            init_e[i, :] = adata.X[init_l == c].mean(0)
            init_ev[i, :] = adata.X[init_l == c].var(0)

    elif initial_clustering_method == "FS":
        ## needs to output to csv first
        # ofn = OPATH + "fs_" + ONAME + ".csv"
        pd.DataFrame(X).to_csv("fs.csv")
        fsom = flowsom("fs.csv", if_fcs=False, if_drop=True, drop_col=["Unnamed: 0"])

        fsom.som_mapping(
            50,  # x_n: e.g. 100, the dimension of expected map
            50,  # y_n: e.g. 100, the dimension of expected map
            fsom.df.shape[1],
            1,  # sigma: e.g 1, the standard deviation of initialized weights
            0.5,  # lr: e.g 0.5, learning rate
            1000,  # batch_size: 1000, iteration times
            tf_str=None,  # string, e.g. hlog', None, etc - the transform algorithm
            if_fcs=False,  # bool, when the the input file is fcs file. If not, it should be a csv file
            # seed = 10, for reproducing
        )
        start = k
        fsom_num_cluster = 0
        while fsom_num_cluster < k:
            # print(nc, start, fsom_nc)
            fsom.meta_clustering(
                AgglomerativeClustering,
                min_n=start,
                max_n=start,
                verbose=False,
                iter_n=10,
            )  # train the meta clustering for cluster in range(40,45)

            fsom.labeling()
            # fsom.bestk # the best number of clusters within the range of (min_n, max_n)
            fsom_class = np.unique(fsom.df["category"])
            fsom_num_cluster = len(fsom_class)
            start += 1

        fsom_labels = np.array(fsom.df["category"])

        i = 0
        init_l = np.zeros(fsom.df.shape[0], dtype=int)
        init_e = np.zeros((len(fsom_class), fsom.df.shape[1]))
        init_ev = np.zeros((len(fsom_class), fsom.df.shape[1]))
        for row in fsom_class:
            init_l[fsom_labels == row] = i
            init_e[i, :] = fsom.df[fsom_labels == row].mean(0)
            init_ev[i, :] = fsom.df[fsom_labels == row].var(0)
            i += 1

        init_e = init_e[:, :-1]
        init_ev = init_ev[:, :-1]

    adata.obs["init_label"] = init_l
    adata.varm[
        "init_exp_centroids"
    ] = (
        init_e.T
    )  ## An expression matrix (PxC) resulting from a clustering method (i.e., Kmeans)
    adata.varm[
        "init_exp_variances"
    ] = (
        init_ev.T
    )  ## An expression variance (daignal) matrix (PxC) resulting from a clustering method

    return adata


def is_non_negative_float(arg: float):
    return isinstance(arg, Number) and arg > 0


def validate_starling_arguments(
    adata: AnnData,
    dist_option: str,
    singlet_prop: float,
    model_cell_size: bool,
    cell_size_col_name: str,
    model_zplane_overlap: bool,
    model_regularizer: float,
    learning_rate: float,
):
    if type(adata) != AnnData:
        raise ValueError(
            f"Argument `adata` must be of type AnnData, received {type(adata)}."
        )

    if adata.shape[0] < 10 or adata.shape[1] < 10:
        raise ValueError(
            f"Argument `adata` shape must be at least (10,10), received {adata.shape}."
        )

    if type(dist_option) != str or dist_option not in ["T", "N"]:
        raise ValueError(
            f"Argument `dist_option` must be either 'T' or 'N', received {dist_option}"
        )

    if not isinstance(singlet_prop, Number) or 0 > singlet_prop > 1:
        raise ValueError(
            f"Argument `singlet_prop` must be a number between 0 and 1, received {singlet_prop}"
        )

    if not type(model_cell_size) == bool:
        raise ValueError(
            f"Argument `model_cell_size` must be boolean, received {type(model_cell_size)}"
        )

    if model_cell_size and cell_size_col_name not in adata.obs:
        raise ValueError(
            f"Argument `cell_size_col_name` must be a valid column in `adata.obs`"
        )

    if not type(model_zplane_overlap) == bool:
        raise ValueError(
            f"Argument `model_zplane_overlap` must be boolean, received {type(model_cell_size)}"
        )

    if not is_non_negative_float(model_regularizer):
        raise ValueError(
            f"Argument `model_regularizer` must be a non-negative number, received {model_regularizer}"
        )

    if not is_non_negative_float(learning_rate):
        raise ValueError(
            f"Argument `learning_rate` must be a non-negative number, received {learning_rate}"
        )


def model_parameters(adata: AnnData, singlet_prop: float) -> Dict[str, np.ndarray]:
    """Return initial model parameters

    :param adata: The sample to be analyzed, with clusters and annotations from :py:func:`init_clustering`
    :param singlet_prop:  The proportion of anticipated segmentation error free cells

    :return: the model parameters
    """

    init_e = adata.varm["init_exp_centroids"].T
    init_v = adata.varm["init_exp_variances"].T
    init_s = adata.uns["init_cell_size_centroids"]
    init_sv = adata.uns["init_cell_size_variances"]

    nc = init_e.shape[0]
    pi = np.ones(nc) / nc
    tau = np.ones((nc, nc))
    tau = tau / tau.sum()

    model_params = {
        "is_pi": np.log(pi + 1e-6),
        "is_tau": np.log(tau + 1e-6),
        "is_delta": np.log([1 - singlet_prop, singlet_prop]),
    }

    model_params["log_mu"] = np.log(init_e + 1e-6)

    init_v[np.isnan(init_v)] = 1e-6
    model_params["log_sigma"] = np.log(init_v + 1e-6)

    if init_s is not None:
        model_params["log_psi"] = np.log(init_s.astype(float) + 1e-6)

        init_sv[np.isnan(init_sv)] = 1e-6
        model_params["log_omega"] = np.log(init_sv.astype(float) ** 0.5 + 1e-6)

    return model_params


def simulate_data(
    Y: torch.Tensor, S: Union[torch.Tensor, None] = None, model_overlap: bool = True
) -> Tuple[torch.tensor]:
    """Use real data to simulate singlets/doublets (equal proportions).
    Return same number of cells as in Y/S, half of them are singlets and another half are doublets

    :param Y: data matrix of shape m x n
    :param S: data matrix of shape m
    :param model_overlap: If cell size is modelled, should STARLING model z-plane overlap

    :return: the simulated data
    """

    sample_size = int(Y.shape[0] / 2)
    idx_singlet = np.random.choice(Y.shape[0], size=sample_size, replace=True)
    Y_singlet = Y[idx_singlet, :]  ## expression

    idx_doublet = [
        np.random.choice(Y.shape[0], size=sample_size),
        np.random.choice(Y.shape[0], size=sample_size),
    ]
    Y_doublet = (Y[idx_doublet[0], :] + Y[idx_doublet[1], :]) / 2.0

    fake_Y = torch.vstack([Y_singlet, Y_doublet])
    fake_label = torch.concat(
        [
            torch.ones(sample_size, dtype=torch.int),
            torch.zeros(sample_size, dtype=torch.int),
        ]
    )

    if S is None:
        return fake_Y, None, fake_label
    else:
        S_singlet = S[idx_singlet]
        if model_overlap:
            dmax = torch.vstack([S[idx_doublet[0]], S[idx_doublet[1]]]).max(0).values
            dsum = S[idx_doublet[0]] + S[idx_doublet[1]]
            rr_dist = torch.distributions.Uniform(
                dmax.type(torch.float64), dsum.type(torch.float64)
            )
            S_doublet = rr_dist.sample()
        else:
            S_doublet = S[idx_doublet[0]] + S[idx_doublet[1]]
        fake_S = torch.hstack([S_singlet, S_doublet])
        return fake_Y, fake_S, fake_label  ## singlet == 1, doublet == 0


def compute_p_y_given_z(Y, Theta, dist_option):  ## singlet case given expressions
    """:return: # of obs x # of cluster matrix - p(y_n | z_n = c)"""

    mu = torch.clamp(torch.exp(torch.clamp(Theta["log_mu"], min=-12, max=14)), min=0)
    sigma = torch.clamp(
        torch.exp(torch.clamp(Theta["log_sigma"], min=-12, max=14)), min=0
    )

    if dist_option == "N":
        dist_Y = torch.distributions.Normal(loc=mu, scale=sigma)
    else:
        dist_Y = torch.distributions.StudentT(df=2, loc=mu, scale=sigma)

    return dist_Y.log_prob(Y.reshape(-1, 1, Y.shape[1])).sum(
        2
    )  # <- sum because IID over G


def compute_p_s_given_z(S, Theta, dist_option):  ## singlet case given cell sizes
    """:return: # of obs x # of cluster matrix - p(s_n | z_n = c)"""

    psi = torch.clamp(torch.exp(torch.clamp(Theta["log_psi"], min=-12, max=14)), min=0)
    omega = torch.clamp(
        torch.exp(torch.clamp(Theta["log_omega"], min=-12, max=14)), min=0
    )

    if dist_option == "N":
        dist_S = torch.distributions.Normal(loc=psi, scale=omega)
    else:
        dist_S = torch.distributions.StudentT(df=2, loc=psi, scale=omega)

    return dist_S.log_prob(S.reshape(-1, 1))


def compute_p_y_given_gamma(Y, Theta, dist_option):  ## doublet case given expressions
    """:return: # of obs x # of cluster x # of cluster matrix - p(y_n | gamma_n = [c,c'])"""

    mu = torch.clamp(torch.exp(torch.clamp(Theta["log_mu"], min=-12, max=14)), min=0)
    sigma = torch.clamp(
        torch.exp(torch.clamp(Theta["log_sigma"], min=-12, max=14)), min=0
    )

    mu2 = mu.reshape(1, mu.shape[0], mu.shape[1])
    mu2 = (mu2 + mu2.permute(1, 0, 2)) / 2.0  # C x C x G matrix

    sigma2 = sigma.reshape(1, mu.shape[0], mu.shape[1])
    sigma2 = (sigma2 + sigma2.permute(1, 0, 2)) / 2.0

    if dist_option == "N":
        dist_Y2 = torch.distributions.Normal(loc=mu2, scale=sigma2)
    else:
        dist_Y2 = torch.distributions.StudentT(df=2, loc=mu2, scale=sigma2)

    return dist_Y2.log_prob(Y.reshape(-1, 1, 1, mu.shape[1])).sum(
        3
    )  # <- sum because IID over G


def compute_p_s_given_gamma(S, Theta, dist_option):  ## singlet case given cell size
    """:return: # of obs x # of cluster x # of cluster matrix - p(s_n | gamma_n = [c,c'])"""

    psi = torch.clamp(torch.exp(torch.clamp(Theta["log_psi"], min=-12, max=14)), min=0)
    omega = torch.clamp(
        torch.exp(torch.clamp(Theta["log_omega"], min=-12, max=14)), min=0
    )  # + 1e-6

    psi2 = psi.reshape(-1, 1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1, 1)
    omega2 = omega2 + omega2.T  # + 1e-6

    if dist_option == "N":
        dist_S2 = torch.distributions.Normal(loc=psi2, scale=omega2)
    else:
        dist_S2 = torch.distributions.StudentT(df=2, loc=psi2, scale=omega2)
    return dist_S2.log_prob(S.reshape(-1, 1, 1))


def compute_p_s_given_gamma_model_overlap(S, Theta):
    """:return: # of obs x # of cluster x # of cluster matrix - p(s_n | gamma_n = [c,c'])"""

    psi = torch.clamp(torch.exp(torch.clamp(Theta["log_psi"], min=-12, max=14)), min=0)
    omega = torch.clamp(
        torch.exp(torch.clamp(Theta["log_omega"], min=-12, max=14)), min=0
    )  # + 1e-6

    psi2 = psi.reshape(-1, 1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1, 1)
    omega2 = omega2 + omega2.T

    ## for v
    ccmax = torch.combinations(psi).max(1).values
    mat = torch.zeros(len(psi), len(psi), dtype=torch.float64).to(DEVICE)
    mat[np.triu_indices(len(psi), 1)] = ccmax
    mat += mat.clone().T
    mat += torch.eye(len(psi)).to(DEVICE) * psi

    ## for s
    c = 1 / (np.sqrt(2) * omega2)
    q = psi2 - S.reshape(-1, 1, 1)
    p = mat - S.reshape(-1, 1, 1)

    const = 1 / (2 * (psi2 - mat))
    lbp = torch.special.erf(p * c)
    ubp = torch.special.erf(q * c)
    prob = torch.clamp(const * (ubp - lbp), min=1e-6, max=1.0)

    return prob.log()


def compute_posteriors(Y, S, Theta, dist_option, model_overlap):
    ## priors
    log_pi = torch.nn.functional.log_softmax(Theta["is_pi"], dim=0)  ## C
    log_tau = torch.nn.functional.log_softmax(
        Theta["is_tau"].reshape(-1), dim=0
    ).reshape(
        log_pi.shape[0], log_pi.shape[0]
    )  ## CxC
    log_delta = torch.nn.functional.log_softmax(Theta["is_delta"], dim=0)  ## 2

    prob_y_given_z = compute_p_y_given_z(
        Y, Theta, dist_option
    )  ## log p(y_n|z=c) -> NxC
    prob_data_given_z_d0 = (
        prob_y_given_z + log_pi
    )  ## log p(y_n|z=c) + log p(z=c) -> NxC + C -> NxC

    if S is not None:
        prob_s_given_z = compute_p_s_given_z(
            S, Theta, dist_option
        )  ## log p(s_n|z=c) -> NxC
        prob_data_given_z_d0 += (
            prob_s_given_z  ## log p(y_n|z=c) + log p(s_n|z=c) -> NxC
        )

    prob_y_given_gamma = compute_p_y_given_gamma(
        Y, Theta, dist_option
    )  ## log p(y_n|g=[c,c']) -> NxCxC
    prob_data_given_gamma_d1 = (
        prob_y_given_gamma + log_tau
    )  ## log p(y_n|g=[c,c']) + log p(g=[c,c']) -> NxCxC

    if S is not None:
        if model_overlap == "Y":
            prob_s_given_gamma = compute_p_s_given_gamma_model_overlap(
                S, Theta
            )  ## log p(s_n|g=[c,c']) -> NxCxC
        else:
            prob_s_given_gamma = compute_p_s_given_gamma(
                S, Theta, dist_option
            )  ## log p(s_n|g=[c,c']) -> NxCxC

        prob_data_given_gamma_d1 += (
            prob_s_given_gamma  ## log p(y_n|g=[c,c']) + log p(s_n|g=[c,c']) -> NxCxC
        )

    prob_data = torch.hstack(
        [
            prob_data_given_z_d0 + log_delta[0],
            prob_data_given_gamma_d1.reshape(Y.shape[0], -1) + log_delta[1],
        ]
    )
    prob_data = torch.logsumexp(prob_data, dim=1)  ## N
    ## log p(data) =
    # case 1:
    # log p(y_n|z=c) + log p(d_n=0) +
    # log p(y_n|g=[c,c']) + log p(d_n=1)
    # case 2:
    # log p(y_n,s_n|z=c) + log p(d_n=0) +
    # log p(y_n,s_n|g=[c,c']) + log p(d_n=1)

    ## average negative likelihood scores
    cost = -prob_data.mean()  ## a value

    ## integrate out z
    prob_data_given_d0 = torch.logsumexp(
        prob_data_given_z_d0, dim=1
    )  ## p(data_n|d=0)_N
    prob_singlet = torch.clamp(
        torch.exp(prob_data_given_d0 + log_delta[0] - prob_data), min=0.0, max=1.0
    )

    ## assignments
    r = prob_data_given_z_d0.T + log_delta[0] - prob_data  ## p(d=0,z=c|data)
    v = (
        prob_data_given_gamma_d1.T + log_delta[1] - prob_data
    )  ## p(d=1,gamma=[c,c']|data)

    return r.T, v.T, cost, prob_singlet


def predict(
    dataLoader: DataLoader,
    model_params: Dict[str, torch.Tensor],
    dist_option: str,
    model_cell_size: bool,
    model_zplane_overlap: bool,
    threshold: float = 0.5,
):
    """return singlet/doublet probabilities, singlet cluster assignment probabilty matrix & assignment labels

    :param dataLoader: the dataloader
    :param model_params: the model parameters
    :param dist_option: str, one of 'T' for Student-T (df=2) or 'N' for Normal (Gaussian)
    :param model_cell_size: bool
    :param model_zplane_overlap: whether z-plane overlap is modeled
    :param threshold:
    :return:
    """

    singlet_prob_list = []
    gamma_assig_prob_list = []
    singlet_assig_prob_list = []
    # singlet_assig_label_list = []

    with torch.no_grad():
        for i, bat in enumerate(dataLoader):
            if model_cell_size:
                # print(bat[0].shape)
                # print(bat[1].shape)
                (
                    singlet_assig_prob,
                    gamma_assig_prob,
                    _,
                    singlet_prob,
                ) = compute_posteriors(
                    bat[0].to(DEVICE),
                    bat[1].to(DEVICE),
                    model_params,
                    dist_option,
                    model_zplane_overlap,
                )
            else:
                (
                    singlet_assig_prob,
                    gamma_assig_prob,
                    _,
                    singlet_prob,
                ) = compute_posteriors(
                    bat.to(DEVICE),
                    None,
                    model_params,
                    dist_option,
                    model_zplane_overlap,
                )

            singlet_prob_list.append(singlet_prob.cpu())
            gamma_assig_prob_list.append(gamma_assig_prob.exp().cpu())
            singlet_assig_prob_list.append(singlet_assig_prob.exp().cpu())

            # batch_pred = singlet_assig_prob.exp().max(1).indices
            # batch_pred[singlet_prob <= threshold] = -1
            # singlet_assig_label_list.append(batch_pred.cpu())

    singlet_prob = torch.cat(singlet_prob_list)
    gamma_assig_prob = torch.cat(gamma_assig_prob_list)
    singlet_assig_prob = torch.cat(singlet_assig_prob_list)
    # singlet_assig_label = torch.cat(singlet_assig_label_list)

    return singlet_prob, singlet_assig_prob, gamma_assig_prob
