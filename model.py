import torch
from torch import nn
import numpy as np


class TensorRankBlock(nn.Module):
    def __init__(self):
        super(TensorRankBlock, self).__init__()

    @staticmethod
    def PSVT(X, tau, r=1):
        X = torch.squeeze(X)
        [U, S, V] = torch.svd(X)
        V = torch.t(V)

        Xd = torch.mm(torch.mm(U[:, 0:r], torch.diag(torch.complex(S[0:r], torch.tensor(0.0).cuda()))), torch.conj(torch.t(torch.t(V)[:, 0:r])))
        diagS = nn.functional.relu(S[r:] - tau)
        diagS = torch.squeeze(diagS[torch.nonzero(diagS)])
        svp = np.prod(list(diagS.shape))

        if svp >= 1:
            Xd = Xd + torch.mm(torch.mm(U[:, r:r + svp], torch.diag(diagS)), torch.conj(torch.t(torch.t(V)[:, r:r + svp])))

        Xd = torch.unsqueeze(Xd, 0)
        return Xd

    def forward(self, x, tau):
        # calculate Psi_hat along the third dimension
        x = x.permute(0, 3, 2, 1) # to channel last
        x_fft = torch.fft.fft(torch.squeeze(x), dim=2)
        # calculate solutions for three frontal slices of X_hat (Eq. 20)
        # using PSVT operator (Eq. 19)
        t_fft = torch.zeros(x_fft.shape, dtype=x_fft.dtype).cuda()
        t_fft[:, :, 0] = self.PSVT(x_fft[:, :, 0], tau)
        t_fft[:, :, 1] = self.PSVT(x_fft[:, :, 1], tau)
        t_fft[:, :, 2] = self.PSVT(x_fft[:, :, 2], tau)
        # inverse FFT of X_hat to create X (Eq. 21)
        t_ifft = torch.unsqueeze(torch.fft.irfft(t_fft, n=3, dim=2), 0)
        t_ifft = t_ifft.permute(0, 3, 2, 1) # to channel first for CNN
        return t_ifft


class ProximalBlock(nn.Module):
    def __init__(self):
        super(ProximalBlock, self).__init__()
        # This network structure is illustrated in Fig. 3
        self.proximal = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        output = self.proximal(x)
        return output


class RPCA_Block(nn.Module):
    def __init__(self):
        super(RPCA_Block, self).__init__()

        self.Proximal_X = TensorRankBlock()  # solution for PSTNN (Eq. 17-21)
        self.Proximal_P = ProximalBlock()    # proximal operator V_k (Eq. 28)
        self.Proximal_Q = ProximalBlock()    # proximal operator W_k (Eq. 29)

        self.lamb  = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.delta = torch.nn.Parameter(torch.tensor(1e-6, dtype=torch.float32), requires_grad=True).cuda()
        self.mu    = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float32), requires_grad=True).cuda()
        self.alpha = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float32), requires_grad=True).cuda()
        self.beta  = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float32), requires_grad=True).cuda()

    def forward(self, X, L1, L2, L3, E, S, P, Q, Omega):

        # update X (Eq. 20-21)
        psi_x = self.mu + self.alpha
        Psi_X = (L1 - L2 + self.mu * Omega - self.mu * E - self.mu * S + self.alpha * P) / psi_x
        X_k = self.Proximal_X(Psi_X, torch.tensor(1.).cuda() / psi_x)

        # update E (Eq. 23)
        psi_e = self.mu + self.beta
        Psi_E = (L1 - L3 + self.mu * Omega - self.mu * X_k - self.mu * S + self.beta * Q) / psi_e
        E_k = torch.mul(torch.sign(Psi_E), nn.functional.relu(torch.abs(Psi_E) - self.lamb / psi_e))

        # update S (Eq. 25)
        # S is T in the paper, S is legacy name
        # mysterious errors might happen, so I leave it be
        Y = Omega - X_k - E_k + L1 / self.mu
        S_k = torch.mul(Y, torch.tensor(1.).cuda() - Omega) + \
            torch.mul(Y, Omega) * torch.min(torch.tensor(1.).cuda(), self.delta / (torch.norm(torch.mul(Y, Omega), 'fro') + 1e-6))

        # update P (Eq. 28)
        # V_k is self.Proximal_P
        P_k = self.Proximal_P(X_k + L2 / (self.alpha + 1e-6))

        # update Q (Eq. 29)
        # W_k is self.Proximal_Q
        Q_k = self.Proximal_Q(E_k + L3 / (self.beta + 1e-6))

        # update Lagrange multipliers (Eq. 30)
        L1_k = L1 + self.mu * (Omega - X_k - E_k - S_k)
        L2_k = L2 + self.alpha * (X_k - P_k)
        L3_k = L3 + self.beta * (E_k - Q_k)

        return X_k, L1_k, L2_k, L3_k, E_k, S_k, P_k, Q_k


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
        X_hdr = self.composer(X_hat.permute(0, 2, 1, 3))
        X_hdr = torch.squeeze(X_hdr)

        return X_hat, X_hdr

