import torch
from torch import nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1_first = nn.Conv2d(in_channels=1,  out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_1     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_3     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_4     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_5     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_6     = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_last  = nn.Conv2d(in_channels=64, out_channels=1,  kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1_first(x)
        out = self.conv3_1(out)
        out = self.relu(out)

        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.relu(out)

        out = self.conv3_4(out)
        out = self.conv3_5(out)
        out = self.relu(out)

        out = self.conv3_6(out)
        out = self.conv1_last(out)

        out += identity
        return out


class ProximalBlock(nn.Module):
    def __init__(self):
        super(ProximalBlock, self).__init__()
        self.proximal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        output = self.proximal(x)
        return output


class RPCA_Block(nn.Module):
    def __init__(self):
        super(RPCA_Block, self).__init__()
        self.Proximal_P = ProximalBlock()
        self.Proximal_Q = ResBlock()

        self.lamb  = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.delta = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.mu    = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.alpha = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.beta  = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()

    def forward(self, X, L1, L2, L3, E, S, P, Q, Omega):

        # update X
        psi_x = self.mu + self.alpha
        Psi_X = torch.tensor(1.).cuda() / psi_x * (L1 - L2 + self.mu * Omega - self.mu * E - self.mu * S + self.alpha * P)
        X_k = self.PSVT(Psi_X, torch.tensor(1.).cuda() / (torch.sqrt(psi_x) + 1e-6))

        # update E
        psi_e = self.mu + self.beta
        Psi_E = torch.tensor(1.).cuda() / psi_e * (L1 - L3 + self.mu * Omega - self.mu * X_k - self.mu * S + self.beta * Q)
        E_k = torch.mul(torch.sign(Psi_E), nn.functional.relu(torch.abs(Psi_E) - self.lamb / (torch.sqrt(psi_e) + 1e-6)))

        # update S
        Y = Omega - X_k - E_k + L1 / self.mu
        S_k = torch.mul(Y, 1. - Omega) + \
            torch.mul(Y, Omega) * min(1., self.delta.item() / (torch.norm(torch.mul(Y, Omega), 'fro') + 1e-6))

        # update P
        P_k = self.Proximal_P(X_k + L2 / (self.alpha + 1e-6))

        # update Q
        Q_k = self.Proximal_Q(E_k + L3 / (self.beta + 1e-6))

        # update Lambda
        L1_k = L1 + self.mu * (Omega - X_k - E_k - S_k)
        L2_k = L2 + self.alpha * (X_k - P_k)
        L3_k = L3 + self.beta * (E_k - Q_k)

        return X_k, L1_k, L2_k, L3_k, E_k, S_k, P_k, Q_k

    @staticmethod
    def PSVT(X, tau, r=1):
        X = torch.squeeze(X)
        [U, S, V] = torch.svd(X)
        V = torch.t(V)

        Xd = torch.mm(torch.mm(U[:, 0:r], torch.diag(S[0:r])), torch.t(torch.t(V)[:, 0:r]))

        diagS = S
        diagS2 = torch.max(diagS[r:] - tau, torch.zeros(diagS[r:].shape).cuda())
        diagS2 = torch.squeeze(diagS2[torch.nonzero(diagS2)])
        svp = np.prod(list(diagS2.shape))

        if svp >= 1:
            Xd = Xd + torch.mm(torch.mm(U[:, r:r + svp], torch.diag(diagS2)), torch.t(torch.t(V)[:, r:r + svp]))

        Xd = torch.unsqueeze(Xd, 0)
        Xd = torch.unsqueeze(Xd, 0)
        return Xd


class RPCA_Net(nn.Module):
    def __init__(self, N_iter):
        super(RPCA_Net, self).__init__()
        self.N_iter = N_iter

        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block())

        self.network = nn.ModuleList(blocks_list)
        self.composer = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, image, Omega):

        OmegaD = torch.mul(image, Omega)
        OmegaD = torch.unsqueeze(OmegaD, 0)

        X  = OmegaD
        L1 = torch.zeros(OmegaD.size()).cuda()
        L2 = torch.zeros(OmegaD.size()).cuda()
        L3 = torch.zeros(OmegaD.size()).cuda()
        E  = torch.zeros(OmegaD.size()).cuda()
        S  = torch.zeros(OmegaD.size()).cuda()
        P  = torch.zeros(OmegaD.size()).cuda()
        Q  = torch.zeros(OmegaD.size()).cuda()

        layers = []
        for i in range(0, self.N_iter):
            [X, L1, L2, L3, E, S, P, Q] = self.network[i](X, L1, L2, L3, E, S, P, Q, OmegaD)
            layers.append(torch.stack([X, L1, L2, L3, E, S, P, Q]))

        X_hat = layers[-1][0]
        X_hdr = self.composer(X_hat.permute(0, 3, 2, 1))
        X_hdr = torch.squeeze(X_hdr)

        return X_hdr

