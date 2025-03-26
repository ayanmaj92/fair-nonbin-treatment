import torch
import torch.nn.functional as F

from .sem_base import SEM


class TreatmentData(SEM):
    def __init__(self, sem_name="dummy", num_sensitive=None,
                 num_covariate=None, num_treatment=None):
        functions = None
        inverses = None

        # AM: Adding
        self.num_sensitive = num_sensitive
        self.num_covariate = num_covariate
        self.num_treatment = num_treatment

        self.total_features = num_sensitive + num_covariate + num_treatment + 1

        if sem_name == "dummy":
            x = lambda *args: args[-1]
            functions = [x for _ in range(self.total_features)]
            inverses = [x for _ in range(self.total_features)]

        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        # adj = torch.zeros((7, 7))
        # sensitive_idxs = range(0, 1)
        # covariate_idxs = range(1, 1 + 3)
        # treatment_idxs = range(1 + 3, 1 + 3 + 2)
        adj = torch.zeros((self.total_features, self.total_features))
        sensitive_idxs = range(0, self.num_sensitive)
        covariate_idxs = range(self.num_sensitive, self.num_sensitive + self.num_covariate)
        treatment_idxs = range(self.num_sensitive + self.num_covariate,
                               self.num_sensitive + self.num_covariate + self.num_treatment)
        tmp = adj[covariate_idxs, :]
        tmp[:, sensitive_idxs] = 1
        adj[covariate_idxs] = tmp
        tmp = adj[covariate_idxs[1:], :]
        tmp[:, covariate_idxs[:-1]] = torch.eye(len(covariate_idxs) - 1)
        adj[covariate_idxs[1:]] = tmp
        tmp = adj[treatment_idxs, :]
        tmp[:, sensitive_idxs] = 1
        tmp[:, covariate_idxs] = 1
        adj[treatment_idxs] = tmp
        tmp = adj[treatment_idxs[1:], :]
        tmp[:, treatment_idxs[:-1]] = torch.eye(len(treatment_idxs) - 1)
        adj[treatment_idxs[1:]] = tmp
        tmp = adj[-1, :]
        tmp[:-1] = 1
        adj[-1, :] = tmp
        # adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        # adj[1, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        # adj[2, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        # adj[3, :] = torch.tensor([1, 1, 1, 0, 0, 0, 0])
        # adj[4, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        # adj[5, :] = torch.tensor([1, 1, 0, 0, 1, 0, 0])
        # adj[6, :] = torch.tensor([1, 1, 0, 0, 1, 1, 0])

        if add_diag:
            adj += torch.eye(self.total_features)

        return adj

    def intervention_index_list(self):
        return list(range(0, self.num_sensitive))
