import os
import sys
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

sys.path.append("../")
import argparse
from ffnn import FFNN

from sampling_utils import lam_sampler
from tqdm import trange
from time import sleep

def compute_NTK(y_pred, net, optimizer, num_data=None):

    num_data = num_data or len(y_pred)

    param_withgrad = [
        param for param in net.parameters() if param.requires_grad is True
    ]  # list of parameters with grad
    device = param_withgrad[0].device
    param_vec = torch.nn.utils.parameters_to_vector(param_withgrad)
    (param_vec_dim,) = param_vec.shape
    features = torch.zeros(num_data, param_vec_dim, requires_grad=False)

    for i in range(num_data):
        optimizer.zero_grad()
        y_pred[i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False).to(device)
        for param in param_withgrad:
            p_grad = torch.cat((p_grad, param.grad.reshape(-1)))
        features[i] = p_grad

    ntk = features @ features.transpose(-2, -1)
    ntk = ntk.cpu().data.numpy()

    return ntk

def run_gd(
    path,
    x,
    y,
    xtest,
    ytest,
    net,
    lam_rvs,
    num_iter,
    lr,
    thinning=1000,
    save_every=1000,
    ntk_num_data=None,
):

    net.freeze_grad_lambdas()
    net.freeze_output_v()

    for n, p in net.named_parameters():
        if p.requires_grad:
            print(n)

    net.init_weights(lam_rvs, sigma_v=np.sqrt(2), sigma_b=0.0)
    p = net.p
    loss = nn.MSELoss()

    results = {
        "step": [],
        "trainrisk": [],
        "testrisk": [],
        "log_lambdas": net.hidden_layers[0].log_lambdas.cpu().data.numpy().reshape(-1),
        "ntk": [],
        "ntk_eig": [],
        "v1diff": [],
        "v1norm": [],
    }

    v1_init = net.hidden_layers[0].v.cpu().data.detach().clone().numpy()

    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    tbar = trange(0, num_iter + 1, desc="Bar desc", leave=True)
    for epoch in tbar:

        y_pred = net(x)
        trainrisk = loss(y_pred, y)

        if epoch % thinning == 0:

            with torch.no_grad():
                ytest_pred = net(xtest)
                testrisk = loss(ytest_pred, ytest)

                line = (
                    f"lr {optimizer.param_groups[0]['lr']:.4f} "
                    f"train risk {trainrisk.data.item():.4f} "
                    f"test risk {testrisk.data.item():.4f} "
                )
                tbar.set_description(line, refresh=True)
                sleep(0.001)

            results["step"].append(epoch)
            results["trainrisk"].append(trainrisk.cpu().data.item())
            results["testrisk"].append(testrisk.cpu().data.item())
            ntk = compute_NTK(y_pred, net, optimizer, ntk_num_data)
            results["ntk"].append(ntk)
            results["ntk_eig"].append(np.linalg.eigh(ntk)[0])

            v1 = net.hidden_layers[0].v.cpu().data.numpy()
            results["v1diff"].append(((v1 - v1_init) ** 2).sum(axis=1))
            results["v1norm"].append((v1 ** 2).sum(axis=1))

        if epoch % save_every == 0:
            torch.save(results, os.path.join(path, "gd.th"))
            torch.save(net.state_dict(), os.path.join(path, "model.th"))

        optimizer.zero_grad()
        trainrisk.backward()
        optimizer.step()

    torch.save(results, os.path.join(path, "gd.th"))
    torch.save(net.state_dict(), os.path.join(path, "model.th"))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--expid", type=str, default="run1")
    parser.add_argument("--num_iter", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--thinning", type=int, default=1000)
    parser.add_argument("--ntk_num_data", type=int, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.a == 1.0:
        lam_rvs = lam_sampler(args.p, "iid")
        model_id = "iid"
    elif args.a == 0.0:
        lam_rvs = lam_sampler(args.p, "zipfian", alpha=args.alpha)
        model_id = f"zipfian-alpha{args.alpha}"
    else:
        lam_rvs = lam_sampler(args.p, "const_and_zipfian", a=args.a, alpha=args.alpha)
        model_id = f"const_and_zipfian-alpha{args.alpha}-a{args.a}"
    model_id += f"-p{args.p}-L{args.L}-lr{args.lr}"
    path = (
        os.path.join("./results", model_id, args.expid)
        if args.path is None
        else args.path
    )
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(args.__dict__, f)

    data = torch.load("data.th")
    x, y = data["x"].cuda(), data["y"].cuda()
    xtest, ytest = data["xtest"].cuda(), data["ytest"].cuda()
    net = FFNN(x.shape[-1], args.L, args.p, 1, bias=False).cuda()

    params = {
        "num_iter": args.num_iter,
        "lr": args.lr,
        "alpha": args.alpha,
        "a": args.a,
        "p": args.p,
        "L": args.L,
        "ntk_num_data": args.ntk_num_data,
    }
    torch.save(params, os.path.join(path, "params.th"))

    results = run_gd(
        path,
        x,
        y,
        xtest,
        ytest,
        net,
        lam_rvs,
        args.num_iter,
        args.lr,
        ntk_num_data=args.ntk_num_data,
    )
