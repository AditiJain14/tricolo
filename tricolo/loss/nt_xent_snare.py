import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logits = logits.reshape(1, -1)
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss


    def forward(self, zis, zjs, labels,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs

        # labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        # masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1, 0, 1)) / temperature

        loss_b = self.softXEnt(labels, logits_ba)

        return loss_b



class NTXentLoss_neg(torch.nn.Module):
    '''
    Source https://github.com/joshr17/HCL/blob/main/image/main.py
    '''
    def __init__(self, device, batch_size, temperature, tau_plus, beta, estimator):
        super(NTXentLoss_neg, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        self.device = device
    
    def get_negative_mask(self):
        negative_mask = torch.ones((self.batch_size, 2 * self.batch_size), dtype=bool)
        for i in range(self.batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + self.batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask
    
    
    def forward(self, out_1, out_2):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        old_neg = neg.clone()
        mask = self.get_negative_mask().to(self.device)
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        if self.estimator=='hard':
            N = self.batch_size * 2 - 2
            imp = (self.beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
        elif self.estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss