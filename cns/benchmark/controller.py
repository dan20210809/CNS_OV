import json
import torch
import numpy as np
from typing import Union, Dict
from ..models.graph_vs import GraphVS
from ..midend.graph_gen import GraphData
from ..ablation.ibvs.ibvs import IBVS
import openvino
from pathlib import Path

import pickle

class OVGraphVS(GraphVS):
    def __init__(self, ckpt_path, device, hidden_dim=128):
        core = openvino.Core()
        
        self.net = core.compile_model(Path(ckpt_path)/"openvino_model.xml","CPU")
        self.hidden_dim = hidden_dim

    def __call__(self, data, hidden):
        inputs = {}
        inputs["x_cur"] = getattr(data, "x_cur")
        inputs["x_tar"] = getattr(data, "x_tar")
        inputs["pos_cur"] = getattr(data, "pos_cur")
        inputs["pos_tar"] = getattr(data, "pos_tar")

        inputs["l1_dense_edge_index_cur"] = getattr(data, "l1_dense_edge_index_cur")
        inputs["l1_dense_edge_index_tar"] = getattr(data, "l1_dense_edge_index_tar")

        inputs["l0_to_l1_edge_index_j_cur"] = getattr(data, "l0_to_l1_edge_index_j_cur")
        inputs["l0_to_l1_edge_index_i_cur"] = getattr(data, "l0_to_l1_edge_index_i_cur")

        inputs["cluster_mask"] = getattr(data, "cluster_mask")
        inputs["cluster_centers_index"] = getattr(data, "cluster_centers_index")
        inputs["num_clusters"] = getattr(data, "num_clusters")
        
        if hidden is not None:
            inputs["new_scene"] = torch.tensor(False)
            inputs["hidden"] = hidden
            torch.set_printoptions(threshold=10_000)

            #dump input for test ov inference
            #dump_file = 'ov_input_data.txt'
            #if Path(dump_file).exists() == False:
            #    file = open(dump_file, 'wb')
            #    pickle.dump(inputs, file)
            #    file.close()
        else:
            inputs["new_scene"] = torch.tensor(True)
            inputs["hidden"] = torch.zeros(getattr(data, "num_clusters").sum(), self.hidden_dim)
            #print("[DEBUG] new scene is True, hidden size={}".format(type(hidden)))
        return self.net(inputs)


class GraphVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        self.device = torch.device(device)
        # self.net: GraphVS = torch.load(ckpt_path, map_location=self.device)["net"]
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if hasattr(ckpt, "net") and isinstance(ckpt["net"], torch.nn.Module):
            self.net: GraphVS = ckpt["net"]
        else:
            self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
            self.net.load_state_dict(ckpt)
        self.net.eval()
        self.hidden = None

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)
            
            if getattr(data, "new_scene").any():
                print("[INFO] Got new scene, set hidden state to zero")
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            # breakpoint()
            self.hidden = raw_pred[-1]
            vel = self.net.postprocess(raw_pred, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel

class OVGraphVSController(object):
    def __init__(self, ckpt_path: str, device="CPU"):
        self.device = torch.device(device)
        ckpt_path="cns_ov"
        self.net = OVGraphVS(ckpt_path, "CPU")

        self.hidden = None

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)
            
            if getattr(data, "new_scene").any():
                print("[INFO] Got new scene, set hidden state to zero")
                self.hidden = None
            
            raw_pred = self.net(data, self.hidden)
            # breakpoint()
            raw_pred_torch = []
            for i in range(len(raw_pred)):
                raw_pred_torch.append(torch.tensor(raw_pred[i]))
            self.hidden = raw_pred_torch[-1]
            vel = self.net.postprocess(raw_pred_torch, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel


class IBVSController(object):
    def __init__(self, config_path: str):
        with open(config_path, "r") as fp:
            use_mean = json.load(fp)["use_mean"]
        self.ibvs = IBVS(use_mean)

    def __call__(self, data: GraphData) -> np.ndarray:
        return self.ibvs(data)


class ImageVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        from ..ablation.ibvs.raft_ibvs import RaftIBVS
        from ..reimpl import ICRA2018, ICRA2021
        
        self.device = torch.device(device)
        self.net: Union[ICRA2018, ICRA2021, RaftIBVS] = \
            torch.load(ckpt_path, map_location=self.device)["net"]
        self.net.eval()
        self.tar_feat = None
    
    def __call__(self, data: Dict) -> np.ndarray:
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            
            if data.get("new_scene", True):
                self.tar_feat = None
            
            data["tar_feat"] = self.tar_feat
            raw_pred = self.net(data)
            self.tar_feat = data["tar_feat"]
            vel = self.net.postprocess(raw_pred, data)
        
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        vel = vel.flatten()
        return vel
