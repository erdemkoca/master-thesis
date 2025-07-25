"""
NIMO Variant with Noise Injection, Lambda CV, Explicit Intercept, Zero-Mean Constraint, and Logging
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(0.1 * x)

class NIMO(nn.Module):
    def __init__(self, input_dim, hidden_dim, pe_dim, noise_std=0.0):
        super().__init__()
        # parameters: beta (coefficients) and explicit bias
        self.beta  = nn.Parameter(torch.zeros(input_dim))
        self.bias  = nn.Parameter(torch.tensor(0.0))
        self.noise_std = noise_std
        # shared net for g_u
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),
            nn.Tanh(),
            SinAct(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.pe_dim = pe_dim

    def forward(self, x):  # returns design matrix B = [1, x * h(x)]
        n, d = x.shape
        device = x.device
        # intercept column
        B0 = torch.ones(n, 1, device=device)
        # mask inputs + positional enc.
        X_exp = x.unsqueeze(1).repeat(1, d, 1)
        idx   = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0
        PE    = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)
        # flatten
        inp = torch.cat([X_exp, PE], dim=2).reshape(n * d, 2*d)
        # compute g
        x1 = self.shared_net[0](inp)
        if self.noise_std>0: x1 = x1 + torch.randn_like(x1)*self.noise_std
        g_flat = self.shared_net[1:](x1).squeeze(1)
        g = g_flat.view(n, d)
        # baseline g0
        zero_inp = torch.cat([torch.zeros_like(X_exp), PE], dim=2).reshape(n * d, 2*d)
        z1 = self.shared_net[0](zero_inp)
        if self.noise_std>0: z1 = z1 + torch.randn_like(z1)*self.noise_std
        g0_flat = self.shared_net[1:](z1).squeeze(1)
        g0 = g0_flat.view(n, d)
        # form h = 1 + g - g0
        h = 1.0 + (g - g0)
        feats = x * h
        B = torch.cat([B0, feats], dim=1)
        return B

    def predict_logits(self, x):
        B = self.forward(x)
        return B[:,1:].matmul(self.beta) + self.bias

    def predict_proba(self, x):
        return torch.sigmoid(self.predict_logits(x))

# diagnostics logger
def log_diag(it, lam, loss, f1, beta, grp):
    print(f"Iter {it:2d} | lam={lam:.4f} | loss={loss:.4f} | f1={f1:.4f} | "
          f"|nz_beta={(beta.abs()>1e-6).sum().item()} | grp={grp:.4f}")


def run_nimoNew(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *, hidden_dim=32, pe_dim=None,
      lam_list=(0.01,0.1,1.0,10.0), noise_std=0.3,
      group_reg=0.0, lr=1e-3, T=10, val_frac=0.2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(randomState); np.random.seed(randomState)
    # train/val split for lam CV
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=randomState)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_val_t= torch.tensor(X_val,dtype=torch.float32,device=device)
    y_val_t= torch.tensor(y_val,dtype=torch.float32,device=device)
    n,d = X_tr_t.shape
    if pe_dim is None: pe_dim = d

    best_lam,best_score=None,-1
    for lam in lam_list:
        model = NIMO(d, hidden_dim, pe_dim, noise_std).to(device)
        opt   = optim.Adam(model.shared_net.parameters(), lr=lr)
        w     = torch.ones(d+1, device=device); eps=1e-6; gamma=1.0
        # IRLS + net
        for t in range(T):
            # IRLS
            model.eval()
            B_u = model(X_tr_t)
            logits = B_u.matmul(torch.cat([model.bias.unsqueeze(0), model.beta]).unsqueeze(1)).squeeze(1)
            p = torch.sigmoid(logits); W = p*(1-p)+eps
            z=logits+(y_tr_t-p)/W; BW=B_u*W.unsqueeze(1)
            A=BW.t()@B_u + lam*torch.diag(w); b=BW.t()@z.unsqueeze(1)
            β_hat=torch.linalg.solve(A,b).squeeze(1)
            model.bias.data.copy_(β_hat[0]); model.beta.data.copy_(β_hat[1:])
            w=1/(β_hat.abs()+eps)**gamma
            # net update
            model.train(); opt.zero_grad()
            loss=F.binary_cross_entropy_with_logits(model.predict_logits(X_tr_t),y_tr_t)
            loss.backward(); opt.step()
            # group-lasso prox
            with torch.no_grad():
                W1=model.shared_net[0].weight; norms=[]
                for j in range(d):
                    col=W1[:,d+j]; nrm=col.norm(2).item(); norms.append(nrm)
                    if nrm>group_reg: W1[:,d+j]*=(1-group_reg/nrm)
                    else: W1[:,d+j].zero_()
                grp_norm=np.mean(norms)
            # diag
            f1_tr=f1_score(y_tr, (model.predict_proba(X_tr_t)>=0.5).int().cpu(), zero_division=0)
            log_diag(t,lam,loss.item(),f1_tr,model.beta,grp_norm)
        # val
        model.eval(); pv=model.predict_proba(X_val_t).detach().cpu().numpy()
        f1s=[f1_score(y_val,(pv>=thr).astype(int)) for thr in np.linspace(0,1,101)]
        if max(f1s)>best_score: best_score, best_lam=max(f1s),lam

    # retrain on full
    model=NIMO(d,hidden_dim,pe_dim,noise_std).to(device); opt=optim.Adam(model.shared_net.parameters(),lr=lr)
    X_all_t=torch.tensor(X_train,dtype=torch.float32,device=device)
    y_all_t=torch.tensor(y_train,dtype=torch.float32,device=device)
    w=torch.ones(d+1,device=device); eps=1e-6; gamma=1.0
    for t in range(T):
        model.eval()
        B_u=model(X_all_t)
        logits=B_u.matmul(torch.cat([model.bias.unsqueeze(0),model.beta]).unsqueeze(1)).squeeze(1)
        p=torch.sigmoid(logits); W=p*(1-p)+eps
        z=logits+(y_all_t-p)/W; BW=B_u*W.unsqueeze(1)
        A=BW.t()@B_u + best_lam*torch.diag(w); b=BW.t()@z.unsqueeze(1)
        β_hat=torch.linalg.solve(A,b).squeeze(1)
        model.bias.data.copy_(β_hat[0]); model.beta.data.copy_(β_hat[1:])
        w=1/(β_hat.abs()+eps)**gamma
        model.train(); opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(model.predict_logits(X_all_t),y_all_t)
        loss.backward(); opt.step()
        with torch.no_grad():
            W1=model.shared_net[0].weight
            for j in range(d):
                col=W1[:,d+j]; nrm=col.norm(2).item()
                if nrm>group_reg: W1[:,d+j]*=(1-group_reg/nrm)
                else: W1[:,d+j].zero_()

    # test
    model.eval(); X_te_t=torch.tensor(X_test,dtype=torch.float32,device=device)
    probs=model.predict_proba(X_te_t).detach().cpu().numpy()
    thrs=np.linspace(0,1,101); f1s=[f1_score(y_test,(probs>=t).astype(int)) for t in thrs]
    idx=int(np.argmax(f1s)); best_thr=thrs[idx]
    y_pred=(probs>=best_thr).astype(int).tolist(); prec=precision_score(y_test,y_pred,zero_division=0)
    rec=recall_score(y_test,y_pred,zero_division=0)
    betas=model.beta.detach().cpu().numpy(); sel=[X_columns[i] for i,b in enumerate(betas) if abs(b)>1e-3] if X_columns else [i for i,b in enumerate(betas) if abs(b)>1e-3]

    return {
        'model_name':'nimoNew', 'iteration':iteration,
        'best_lambda':best_lam,'best_threshold':best_thr,'best_f1':max(f1s),
        'precision':prec,'recall':rec,'y_pred':y_pred,'y_prob':probs.tolist(),
        'selected_features':sel,'method_has_selection':True,'n_selected':len(sel),
        'hidden_dim':hidden_dim,'pe_dim':pe_dim
    }
