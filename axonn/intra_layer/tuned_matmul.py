import torch
from functools import partial


TRIALS = 20
WARMUP = 5

def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(f"Matmul tuner: {msg}")

def time_sth(operation):
    torch.cuda.synchronize()

    for _ in range(5):
        operation()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    st.record()
    for _ in range(TRIALS):
        operation()
    en.record()
    
    torch.cuda.synchronize()
    return st.elapsed_time(en)

def matmul_wrapper(A, B, transpose=False, tct = None):
   
    if tct is not None:
        if tct[0]:
            A = A.t().contiguous().t()
        else: 
            A = A.contiguous()

        if tct[1]:
            B = B.t().contiguous().t()
        else:
            B = B.contiguous()

    C = torch.mm(A, B)
    if transpose:
        C = C.t().contiguous()
    return C


def tune(A, B, transpose=False):
    times = {}
    default_op = partial(matmul_wrapper, A, B, transpose, None)
    times["default"] = time_sth(default_op)

    #NN
    nn_op = partial(matmul_wrapper, A, B, transpose, tct=[False, False])
    times["nn"] = time_sth(nn_op)

    #NT
    nn_op = partial(matmul_wrapper, A, B, transpose, tct=[True, False])
    times["nt"] = time_sth(nn_op)

    #TN
    nn_op = partial(matmul_wrapper, A, B, transpose, tct=[False, True])
    times["tn"] = time_sth(nn_op)

    return times

decision_cache = {}


@torch.no_grad()
def tuned_matmul(A, B, signature):
    assert A.ndim == 2
    assert B.ndim == 2
    if signature not in decision_cache:
        print_rank0(f"Entry not found for {signature}.. tuning ...")
        t1 = tune(A, B)
        t2 = tune(B.t(), A.t())
        
        k1 = min(t1, key=t1.get)
        k2 = min(t2, key=t2.get)

        v1, v2 = t1[k1], t2[k2]
        if v1 < v2:
            decision_cache[signature] = (False, k1)
        else:
            decision_cache[signature] = (True,  k2)


    transpose, method = decision_cache[signature]
    assert method in ["default", "nn", "nt", "tn"]
    operands = [A, B] if not transpose else [B.t(), A.t()]
    
    if method == "default":
        answer = torch.mm(*operands)
        if transpose:
            answer = answer.t().contiguous()
        return answer

    if method == "nn":
        tct = [False, False]
    
    elif method == "nt":
        tct = [True, False]

    elif method == "tn":
        tct = [False, True]

    return matmul_wrapper(*operands, transpose, tct)
    

   





