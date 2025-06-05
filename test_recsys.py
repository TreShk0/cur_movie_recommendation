"""
recsys_compare_gpu_metrics_v5.py
— расширение v4: добавлены MAP@K, MAR@K, Coverage, Personalization
— для off-line-оценки берём случайных пользователей (>50 оценок),
  скрываем 20 % оценок → ground-truth
— результаты CUR / SVD пишутся в cur_results.csv, svd_results.csv
"""

import csv, time, random
import numpy as np
import pandas as pd
import cupy as cp
import torch

# ---------- настройки ----------
MATRIX_CSV   = "UI_data_1.csv"
DEVICE       = "cuda"
FRACTIONS    = [0.01,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
TOP_K        = 10
MIN_SCORE    = 1.0
N_EVAL_USERS = 100
TEST_RATIO   = 0.20
SEED         = 42
# --------------------------------

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ---------- служебные функции (скопированы из v4) ----------
def extract_sparse_columns(A_sparse, col_indices, n_rows, r):
    idx,val = A_sparse._indices(),A_sparse._values()
    mask = torch.zeros(idx.shape[1],dtype=torch.bool,device=A_sparse.device)
    for ci in col_indices.cpu(): mask |= (idx[1]==ci.item())
    new_idx,new_val = idx[:,mask].clone(),val[mask].clone()
    mapping = -torch.ones(A_sparse.shape[1],dtype=torch.long,device=A_sparse.device)
    mapping[col_indices] = torch.arange(r,device=A_sparse.device)
    new_idx[1] = mapping[new_idx[1]]
    return torch.sparse_coo_tensor(new_idx,new_val,size=(n_rows,r)).coalesce()

def maxvol_gpu(A,e=1.05,k=100):
    n,r = A.shape
    LU,piv = torch.lu(A)
    P,L,U = torch.lu_unpack(LU,piv)
    I = torch.argmax(P[:,:r],dim=0).clone()
    Q = torch.linalg.solve_triangular(U.T,A.T,upper=False,left=True)
    Y = torch.linalg.solve_triangular(L[:r,:].T,Q,upper=True,left=True,unitriangular=True)
    B = Y.T
    for _ in range(k):
        idx = torch.argmax(torch.abs(B))
        i,j = divmod(idx.item(),r)
        if abs(B[i,j])<=e: break
        I[j]=i
        bj,bi = B[:,j].clone(),B[i,:].clone(); bi[j]-=1
        B -= torch.outer(bj,bi)/B[i,j]
    return I,B

def maxvol_sparse_gpu(A_sparse,r,e=1.05,k=100):
    n,m = A_sparse.shape
    idx,val = A_sparse._indices(),A_sparse._values()
    col_norms = torch.zeros(m,device=A_sparse.device)
    col_norms = col_norms.scatter_add(0,idx[1],val**2).sqrt()
    _,J = torch.topk(col_norms,r); J,_ = torch.sort(J)
    Acols = extract_sparse_columns(A_sparse,J,n,r).to_dense()
    I,_  = maxvol_gpu(Acols,e,k)
    return Acols[I,:],J.cpu().numpy(),I.cpu().numpy()   # W, cols_idx, rows_idx

def cur_predict(uvec,cols_idx,UR,film2id,id2film,known,mu,
                top_k=TOP_K,min_score=MIN_SCORE):
    c = uvec[cols_idx]
    pred = c @ UR + mu
    for t in known:
        if t in film2id: pred[film2id[t]]=-1e9
    top = np.argsort(pred)[::-1]
    recs=[(id2film[i],float(pred[i])) for i in top
          if pred[i]>=min_score and not np.isnan(pred[i])]
    return recs[:top_k]

def svd_predict(uvec,Vt_k,S_k,mu,titles,known,
                top_k=TOP_K,min_score=MIN_SCORE):
    rc = uvec.copy(); m = rc!=0; rc[m]-=mu
    s = np.diag(S_k); s_inv=np.where(s>1e-8,1/s,0)
    u = (rc @ Vt_k.T) * s_inv
    pred = u @ S_k @ Vt_k + mu
    recs=[(t,float(p)) for t,p in zip(titles,pred)
          if t not in known and p>=min_score and not np.isnan(p)]
    recs.sort(key=lambda x:x[1],reverse=True)
    return recs[:top_k]

# ---------- метрики ----------
def apk(gt,pred,k):
    if len(pred)>k: pred=pred[:k]
    hits=score=0.0
    for i,p in enumerate(pred,1):
        if p in gt:
            hits+=1; score+=hits/i
    return score/max(1,min(len(gt),k))

def recall_k(gt,pred,k):
    if len(pred)>k: pred=pred[:k]
    return sum(p in gt for p in pred)/max(1,len(gt))
def calc_metrics(recs,gts,catalog):
    users=list(recs); N=len(users)
    MAP = np.mean([apk(gts[u],recs[u],TOP_K) for u in users])
    MAR = np.mean([recall_k(gts[u],recs[u],TOP_K) for u in users])
    covered={item for u in users for item in recs[u][:TOP_K]}
    coverage=len(covered)/catalog
    if N>1:
        sim=[len(set(recs[u])&set(recs[v]))/TOP_K
             for i,u in enumerate(users) for v in users[i+1:]]
        personalization=1-np.mean(sim)
    else: personalization=0.0
    return MAP,MAR,coverage,personalization

# ---------- main ----------
def main():
    df = pd.read_csv(MATRIX_CSV,index_col=0, nrows=70_000).fillna(0).astype(np.float32)
    df = df.loc[:,(df!=0).any()]
    vals=df.values; titles=df.columns.tolist()
    n_users,n_movies = vals.shape
    mu = vals[vals>0].mean()
    centered = vals.copy(); centered[vals>0]-=mu
    fro = np.linalg.norm(vals)

    # SVD (GPU)
    U,s,Vt = cp.linalg.svd(cp.asarray(centered),full_matrices=False)

    # Sparse (GPU)
    rows,cols = np.nonzero(vals)
    A_sparse = torch.sparse_coo_tensor(
        torch.tensor([rows,cols],device=DEVICE),
        torch.tensor(vals[rows,cols],dtype=torch.float32,device=DEVICE),
        size=(n_users,n_movies)).coalesce()

    # пользователи для off-line оценки
    eligible = np.where((df!=0).sum(axis=1).values>50)[0]
    sample = rng.choice(eligible,size=min(N_EVAL_USERS,len(eligible)),replace=False)

    cur_w=csv.writer(open("cur_results.csv","w",newline="",encoding="utf-8"))
    svd_w=csv.writer(open("svd_results.csv","w",newline="",encoding="utf-8"))
    hdr=["rank","error","time_sec",f"MAP@{TOP_K}",f"MAR@{TOP_K}","Coverage","Personalization"]
    cur_w.writerow(hdr); svd_w.writerow(hdr)

    max_rank=min(n_users,n_movies)
    film2id={t:i for i,t in enumerate(titles)}
    id2film={i:t for t,i in film2id.items()}

    for frac in FRACTIONS:
        r=max(1,round(frac*max_rank))

        # ---------- CUR ----------
        t0=time.time()
        W,cols_idx,rows_idx = maxvol_sparse_gpu(A_sparse,r)
        C = centered[:,cols_idx]             # n×r
        R = centered[rows_idx,:]             # r×m
        Ucur = np.linalg.pinv(W.cpu().numpy())       # r×r (W⁻¹)
        UR   = Ucur @ R                      # r×m
        approx_cur = C @ UR + mu
        err_cur = np.linalg.norm(approx_cur-vals)/fro
        t_cur=time.time()-t0

        # ---------- SVD ----------
        t0=time.time()
        k=min(r,len(s))
        approx_svd = (U[:,:k]*(s[:k])) @ Vt[:k,:] + mu
        err_svd = np.linalg.norm(approx_svd.get()-vals)/fro
        t_svd=time.time()-t0

        # ---------- off-line метрики ----------
        rec_cur,rec_svd,gt = {},{},{}
        for u in sample:
            rated=np.where(vals[u]>0)[0]
            n_test=max(1,int(TEST_RATIO*len(rated)))
            test=rng.choice(rated,size=n_test,replace=False)
            train=np.setdiff1d(rated,test,assume_unique=True)

            gt[u]={id2film[i] for i in test}
            vec=np.zeros(n_movies,np.float32); vec[train]=vals[u,train]
            known={id2film[i] for i in train}

            rec_cur[u]=[t for t,_ in cur_predict(vec,cols_idx,UR,film2id,id2film,
                                                 known,mu,TOP_K,MIN_SCORE)]
            rec_svd[u]=[t for t,_ in svd_predict(vec,Vt[:k,:].get(),
                                                 np.diag(s[:k].get()),mu,
                                                 titles,known,TOP_K,MIN_SCORE)]

        mcur = calc_metrics(rec_cur,gt,n_movies)
        msvd = calc_metrics(rec_svd,gt,n_movies)

        cur_w.writerow([r,f"{err_cur:.6f}",f"{t_cur:.2f}",
                        *[f"{x:.4f}" for x in mcur]])
        svd_w.writerow([r,f"{err_svd:.6f}",f"{t_svd:.2f}",
                        *[f"{x:.4f}" for x in msvd]])

        print(f"[rank {r}]  CUR MAP={mcur[0]:.3f}  SVD MAP={msvd[0]:.3f}")

if __name__ == "__main__":
    main()