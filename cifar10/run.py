import os
import sys
from types import SimpleNamespace

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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision import models, transforms

import json




def compute_NTK(y_pred, net, optimizer, num_data, num_outputs):

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
    dataloaders,
    net: FFNN,
    feature_net, 
    lam_rvs,
    num_iter,
    lr,
    thinning=100,
    save_every=1,
    ntk_num_data=100,
    ntk_num_outputs=10,
):

    net.freeze_grad_lambdas() # Reput
    net.freeze_output_v() # Reput

    for n, p in net.named_parameters():
        if p.requires_grad:
            print(n)

    net.init_weights(lam_rvs, sigma_v=np.sqrt(2), sigma_b=0.0) # Reput
    p = net.p
    loss = nn.CrossEntropyLoss()

    results = {
        "epoch": [],
        "step": [],
        "thinning": thinning,
        "lr": lr,
        "trainingrisk": [],
        "trainingacc": [],
        "testrisk": [],
        "testacc": [],
        "log_lambdas": net.hidden_layers[0].log_lambdas.cpu().data.numpy().reshape(-1), # Reput
        "ntk": [],
        "ntk_eig": [],
        "v1diff": [],
        "v1norm": [],
    }
    results = SimpleNamespace(**results)

    # Reput
    v1_init = net.hidden_layers[0].v.cpu().data.detach().clone().numpy()

    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, num_iter, eta_min=0.1, last_epoch=int(num_iter * 0.8)
    # )
    tbar = trange(1, num_iter + 1, desc="Bar desc", leave=True)
    i = -1
    for epoch in tbar:
        running_loss = 0.0
        running_corrects = 0
        num_images = 0
        
        for inputs, labels in dataloaders['train']:
            net.train()
            i += 1
            
            if i in (801, 4001, 10000, 20000):
                thinning *= 2
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(feature_net(inputs))
            trainingrisk = loss(outputs, labels)
            trainingacc = (outputs.argmax(-1) == labels).float().mean()
            
            optimizer.zero_grad()
            trainingrisk.backward()
            optimizer.step()
            
            running_loss += trainingrisk.data.item() * inputs.size(0)
            running_corrects += trainingacc.data.item()* inputs.size(0)
            num_images += inputs.size(0)
            
            if i % thinning == 0:
                test_running_loss = 0.0
                test_running_corrects = 0
                test_num_images = 0
                with torch.no_grad():
                    net.eval()
                    for inputs, labels in dataloaders['validation']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
            
                        outputs = net(feature_net(inputs))
                        testrisk = loss(outputs, labels)
                        testacc = (outputs.argmax(-1) == labels).float().mean()
                        test_running_loss += testrisk.data.item() * inputs.size(0)
                        test_running_corrects += testacc.data.item() * inputs.size(0)
                        test_num_images += inputs.size(0)
                        
                    line = (
                        f"lr {optimizer.param_groups[0]['lr']:.4f} "
                        f"train risk {running_loss/num_images:.4f} "
                        f"acc {running_corrects/num_images:.4f}| "
                        f"test risk {test_running_loss/test_num_images:.4f} "
                        f"acc {test_running_corrects/test_num_images:.4f}"
                    )
                    tbar.set_description(line, refresh=True)
                    sleep(0.001)
                
                    results.epoch.append(epoch)
                    results.step.append(i)
                    results.trainingrisk.append(running_loss/num_images)
                    results.trainingacc.append(running_corrects/num_images)
                    results.testrisk.append(test_running_loss/test_num_images)
                    results.testacc.append(test_running_corrects/test_num_images)

                    # Reput
                    v1 = net.hidden_layers[0].v.cpu().data.numpy()
                    results.v1diff.append(((v1 - v1_init) ** 2).sum(axis=1))
                    results.v1norm.append((v1 ** 2).sum(axis=1))

        if epoch % save_every == 0:
            torch.save(results, os.path.join(path, "results.th"))
            torch.save(net.state_dict(), os.path.join(path, "model.th"))

        # print('Epoch {}/{}'.format(epoch, num_iter))
        # print('-' * 10)
        
        # print((
        #         f"lr {optimizer.param_groups[0]['lr']:.4f} \n"
        #         f"train risk {running_loss/num_images:.4f} "
        #         f"acc {running_corrects/num_images:.4f}| \n"
        #         f"test risk {test_running_loss/test_num_images:.4f} "
        #         f"acc {test_running_corrects/test_num_images:.4f}"
        #     )
        # )
    torch.save(results, os.path.join(path, "results.th"))
    torch.save(net.state_dict(), os.path.join(path, "model.th"))

    return results

def test_pruning(
    path,
    dataloaders,
    net: FFNN,
    feature_net,
):
    
    loss = nn.CrossEntropyLoss()
    
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
    
    thinning = int(net.p * 0.05)
    tbar = trange(net.p, desc="Prunability:", leave=True)
    for i in tbar:
        net.hidden_layers[0].log_lambdas.data[order[i]] = -torch.inf
        if (i + 1) % thinning == 0:
            with torch.no_grad():
                running_loss = 0.0
                running_corrects = 0
                num_images = 0
                for inputs, labels in dataloaders['train']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = net(feature_net(inputs))
                    trainingrisk = loss(outputs, labels)
                    trainingacc = (outputs.argmax(-1) == labels).float().mean()
                    
                    running_loss += trainingrisk.data.item() * inputs.size(0)
                    running_corrects += trainingacc.data.item()* inputs.size(0)
                    num_images += inputs.size(0)
                    
                test_running_loss = 0.0
                test_running_corrects = 0
                test_num_images = 0
                for inputs, labels in dataloaders['validation']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    outputs = net(feature_net(inputs))
                    testrisk = loss(outputs, labels)
                    testacc = (outputs.argmax(-1) == labels).float().mean()
                    test_running_loss += testrisk.data.item() * inputs.size(0)
                    test_running_corrects += testacc.data.item() * inputs.size(0)
                    test_num_images += inputs.size(0)
                        

                trainrisk = (running_loss/num_images)
                trainacc = (
                    (running_corrects/num_images)
                )
                testrisk = (test_running_loss/test_num_images)
                testacc = (
                    (test_running_corrects/test_num_images)
                )

            
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--expid", type=str, default="run1")
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5)
    #parser.add_argument("--model", type=str, default="const_and_zipfian")
    parser.add_argument(
        "--mode", type=str, default="prune", choices=["train", "prune"]
    )
    parser.add_argument("--model", type=str, default="iid")
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ntk_num_data", type=int, default=50)
    parser.add_argument("--ntk_num_outputs", type=int, default=10)
    parser.add_argument("--thinning", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--a", type=float, default=0.5)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Load and preprocess data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    bs = 64
    image_datasets = {}
    image_datasets['train'] = torchvision.datasets.CIFAR10(root="../../../data", train=True,
                                            download=True, transform=data_transforms['train'])
    image_datasets['validation'] = torchvision.datasets.CIFAR10(root="../../../data", train=False,
                                            download=True, transform=data_transforms['validation'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'validation']}
    

    num_classes = 10
    
    # Load pretrained model for feature extraction
    os.environ['TORCH_HOME'] = '/mnt/disk1/fadhel/pretrained_models/' 
    model = models.resnet18(pretrained=True).to(device)
    
    input_size = model.fc.in_features
    
    model.fc = nn.Identity()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False  

    model.eval()
    

    net = FFNN(input_size, args.L, args.p, num_classes, bias=False).cuda()
    #net = nn.Sequential(
    #    nn.Linear(input_size, args.p),
    #    nn.ReLU(),
    #    nn.Linear(args.p, num_classes)
    #).to(device)
    #net.p = args.p

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

    print(model_id)
    
    path = (
        os.path.join("./results", model_id, args.expid)
    )
    os.makedirs(path, exist_ok=True)

    params = {
        "num_train": args.num_train,
        "num_test": args.num_test,
        "num_iter": args.num_iter,
        "lr": args.lr,
        "model": args.model,
        "p": args.p,
        "L": args.L,
        "alpha": args.alpha,
        "a": args.a,
    }
    
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(args.__dict__, f)

    
    if args.mode == "train":
        results = run_gd(
            path,
            dataloaders,
            net,
            model,
            lam_rvs,
            args.num_iter,
            args.lr,
            thinning=args.thinning,
            ntk_num_data=args.ntk_num_data,
            ntk_num_outputs=args.ntk_num_outputs,
        )
    elif args.mode == "prune":
        print("Pruning")
        net.load_state_dict(torch.load(os.path.join(path, "model.th")))
        test_pruning(
            path,
            dataloaders,
            net,
            model
        )
