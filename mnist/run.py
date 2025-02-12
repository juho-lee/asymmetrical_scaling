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

from fast_nist import FastMNIST
from sampling_utils import lam_sampler
from tqdm import trange
from time import sleep


def compute_NTK(y_pred, net, optimizer, num_data=50, num_outputs=1):

    param_withgrad = [
        param for param in net.parameters() if param.requires_grad is True
    ]  # list of parameters with grad
    device = param_withgrad[0].device
    param_vec = torch.nn.utils.parameters_to_vector(param_withgrad)
    (param_vec_dim,) = param_vec.shape
    features = torch.zeros(num_outputs, num_data, param_vec_dim, requires_grad=False)

    for i in range(num_outputs):
        for j in range(num_data):
            optimizer.zero_grad()
            y_pred[j, i].backward(retain_graph=True)
            p_grad = torch.tensor([], requires_grad=False).to(device)
            for param in param_withgrad:
                p_grad = torch.cat((p_grad, param.grad.reshape(-1)))
            features[i, j] = p_grad

    ntk = features @ features.transpose(-2, -1)
    ntk = ntk.cpu().data.numpy()

    return ntk


def run_gd(
    path,
    x,
    y,
    xtest,
    ytest,
    net: FFNN,
    lam_rvs,
    num_iter,
    lr,
    thinning=1000,
    save_every=10000,
    ntk_num_data=50,
    ntk_num_outputs=1,
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
        "trainacc": [],
        "testrisk": [],
        "testacc": [],
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
        trainacc = (y_pred.argmax(-1) == y.argmax(-1)).float().mean()

        if epoch % thinning == 0:

            with torch.no_grad():
                ytest_pred = net(xtest)
                testrisk = loss(ytest_pred, ytest)
                testacc = (ytest_pred.argmax(-1) == ytest.argmax(-1)).float().mean()

                line = (
                    f"lr {optimizer.param_groups[0]['lr']:.4f} "
                    f"train risk {trainrisk.data.item():.4f} "
                    f"acc {trainacc.data.item():.4f}| "
                    f"test risk {testrisk.data.item():.4f} "
                    f"acc {testacc.data.item():.4f}"
                )
                tbar.set_description(line, refresh=True)
                sleep(0.001)

            results["step"].append(epoch)
            results["trainrisk"].append(trainrisk.cpu().data.item())
            results["trainacc"].append(trainacc.cpu().data.item())
            results["testrisk"].append(testrisk.cpu().data.item())
            results["testacc"].append(testacc.cpu().data.item())
            ntk = compute_NTK(
                y_pred,
                net,
                optimizer,
                num_data=ntk_num_data,
                num_outputs=ntk_num_outputs,
            )
            results["ntk"].append(ntk)
            ntk_eig = []
            for i in range(ntk_num_outputs):
                ntk_eig.append(np.linalg.eigh(ntk[i].squeeze())[0])
            results["ntk_eig"].append(np.stack(ntk_eig))

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


def test_pruning(path, x, y, xtest, ytest, net):
    feature_weights = torch.sum(
        net.output_layer.v.pow(2) * net.hidden_layers[-1].log_lambdas.exp(), 0
    )
    print(feature_weights.shape)
    order = feature_weights.argsort()
    loss = nn.MSELoss()
    results = {
        "num_pruned": [],
        "trainrisk": [],
        "trainacc": [],
        "testrisk": [],
        "testacc": [],
    }

    feature_weights = torch.sum(
        net.output_layer.v.pow(2) * net.hidden_layers[0].log_lambdas.exp(), 0
    )
    order = feature_weights.argsort()
    loss = nn.MSELoss()
    thinning = int(net.p * 0.05)
    tbar = trange(net.p, desc="Prunability:", leave=True)
    for i in tbar:
        net.hidden_layers[0].log_lambdas.data[order[i]] = -torch.inf
        with torch.no_grad():
            ypred = net(x)
            trainrisk = loss(y, ypred).cpu().data.item()
            trainacc = (
                (y.argmax(-1) == ypred.argmax(-1)).float().mean().cpu().data.item()
            )
            ypredtest = net(xtest)
            testrisk = loss(ytest, ypredtest).cpu().data.item()
            testacc = (
                (ytest.argmax(-1) == ypredtest.argmax(-1))
                .float()
                .mean()
                .cpu()
                .data.item()
            )

        if (i + 1) % thinning == 0:
            line = (
                f"Pruned {i+1} nodes, "
                f"train risk {trainrisk:.4f} "
                f"acc {trainacc:.4f}| "
                f"test risk {testrisk:.4f} "
                f"acc {testacc:.4f}"
            )
            tbar.set_description(line, refresh=True)
            sleep(0.001)

            results["num_pruned"].append(i + 1)
            results["trainrisk"].append(trainrisk)
            results["trainacc"].append(trainacc)
            results["testrisk"].append(testrisk)
            results["testacc"].append(testacc)

    torch.save(results, os.path.join(path, "prune.th"))


def test_transfer(
    path,
    xtrans,
    ytrans,
    xtranstest,
    ytranstest,
    net,
    max_trans_p=100,
    trans_p_unit=10,
    num_iters=10000,
    hidden_size=128,
    lr=1.0,
):
    num_classes = ytrans.shape[-1]
    feature_weights = torch.sum(
        -net.output_layer.v.pow(2) * net.hidden_layers[-1].log_lambdas.exp(), 0
    )
    order = feature_weights.argsort()
    accum = {}

    def feature_hook(model, input, output):
        accum["v0"] = output.detach()

    with torch.no_grad():
        net.hidden_layers[0].register_forward_hook(feature_hook)
        net(xtrans)
        feat_all = accum["v0"]
        net(xtranstest)
        feattest_all = accum["v0"]

    results = {
        "feat_dim": [],
        "trainrisk": [],
        "trainacc": [],
        "testrisk": [],
        "testacc": [],
    }

    for feat_dim in range(trans_p_unit, max_trans_p + 1, trans_p_unit):

        feat = feat_all[:, order[:feat_dim]]
        feattest = feattest_all[:, order[:feat_dim]]
        tnet = nn.Sequential(
            nn.Linear(feat.shape[-1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        ).cuda()

        optimizer = torch.optim.SGD(tnet.parameters(), lr=lr)
        loss = nn.MSELoss()
        for i in trange(1, num_iters + 1):
            ypred = tnet(feat)
            trainrisk = loss(ypred, ytrans)
            optimizer.zero_grad()
            trainrisk.backward()
            optimizer.step()

        with torch.no_grad():
            ypred = tnet(feat)
            trainrisk = loss(ypred, ytrans).cpu().data.item()
            trainacc = (
                (ypred.argmax(-1) == ytrans.argmax(-1)).float().mean().cpu().data.item()
            )
            ypredtest = tnet(feattest)
            testrisk = loss(ypredtest, ytranstest).cpu().data.item()
            testacc = (
                (ypredtest.argmax(-1) == ytranstest.argmax(-1))
                .float()
                .mean()
                .cpu()
                .data.item()
            )

        results["feat_dim"].append(feat_dim)
        results["trainrisk"].append(trainrisk)
        results["trainacc"].append(trainacc)
        results["testrisk"].append(testrisk)
        results["testacc"].append(testacc)

        line = (
            f"transfer dim {feat_dim} "
            f"train risk {trainrisk:.4f} "
            f"acc {trainacc:.4f}| "
            f"test risk {testrisk:.4f} "
            f"acc {testacc:.4f}"
        )
        print(line)

    torch.save(results, os.path.join(path, "transfer.th"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "prune", "transfer"]
    )
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--expid", type=str, default="run1")
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_trans_train", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--num_trans_test", type=int, default=5000)
    parser.add_argument("--num_iter", type=int, default=200000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ntk_num_data", type=int, default=50)
    parser.add_argument("--ntk_num_outputs", type=int, default=1)
    parser.add_argument("--thinning", type=int, default=1000)

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
    model_id += f"-p{args.p}-L{args.L}-lr{args.lr}-n{args.num_train}"

    path = (
        os.path.join("./results", model_id, args.expid)
        if args.path is None
        else args.path
    )
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(args.__dict__, f)

    train_dataset = FastMNIST("../data", train=True, download=True, device="cuda")
    test_dataset = FastMNIST("../data", train=False, download=True, device="cuda")
    num_classes = 10

    x, y = train_dataset.data[: args.num_train], train_dataset.targets[: args.num_train]
    x = x.flatten(start_dim=1)
    y = F.one_hot(y, num_classes).float()
    xtest, ytest = (
        test_dataset.data[: args.num_test],
        test_dataset.targets[: args.num_test],
    )
    xtest = xtest.flatten(start_dim=1)
    ytest = F.one_hot(ytest, num_classes).float()
    net = FFNN(x.shape[-1], args.L, args.p, num_classes, bias=False).cuda()

    if args.mode == "train":
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
            ntk_num_outputs=args.ntk_num_outputs,
        )
    elif args.mode == "prune":
        net.load_state_dict(torch.load(os.path.join(path, "model.th")))
        test_pruning(path, x, y, xtest, ytest, net)
    else:
        xtrans, ytrans = (
            train_dataset.data[args.num_train : args.num_train + args.num_trans_train],
            train_dataset.targets[
                args.num_train : args.num_train + args.num_trans_train
            ],
        )
        xtrans = xtrans.flatten(start_dim=1)
        ytrans = F.one_hot(ytrans, num_classes).float()

        xtranstest, ytranstest = (
            test_dataset.data[args.num_test : args.num_test + args.num_trans_test],
            test_dataset.targets[args.num_test : args.num_test + args.num_trans_test],
        )
        xtranstest = xtranstest.flatten(start_dim=1)
        ytranstest = F.one_hot(ytranstest, num_classes).float()
        net.load_state_dict(torch.load(os.path.join(path, "model.th")))
        test_transfer(path, xtrans, ytrans, xtranstest, ytranstest, net)
