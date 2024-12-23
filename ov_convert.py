import openvino
import torch
from cns.benchmark.controller import GraphVSController
from cns.models.graph_vs import GraphVS
from cns.midend.graph_gen import GraphData
import numpy as np

class OVGraphVS(GraphVS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_cur, x_tar, pos_cur, pos_tar, l1_dense_edge_index_cur, l1_dense_edge_index_tar, l0_to_l1_edge_index_j_cur, l0_to_l1_edge_index_i_cur, cluster_mask, cluster_centers_index, num_clusters, magic_number, hidden=None, batch=None):
        l0_to_l1_edge_index_cur = torch.stack([l0_to_l1_edge_index_j_cur,
                                               l0_to_l1_edge_index_i_cur], dim=0)

        if batch is None:
            batch = torch.zeros(x_cur.size(0)).long().to(x_cur.device)
        
        x_clu = self.encoder(
            x_cur, x_tar, pos_cur, pos_tar, cluster_mask, 
            l0_to_l1_edge_index_cur, cluster_centers_index)
        pos_clu = pos_tar[cluster_centers_index]
        batch_clu = batch[cluster_centers_index]
        xx = self.init_hidden(num_clusters.sum()).to(x_cur)

        hidden = torch.where(magic_number<0, xx,hidden)
        
        hidden, x_clu = self.backbone(
            hidden, x_clu, pos_clu, l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu)
        
        vel_si_vec, vel_si_norm = self.decoder(x_clu, cluster_mask, batch_clu)
        
        return vel_si_vec, vel_si_norm, hidden

example_input = {
    "x_cur": torch.rand(511, 2),
    "x_tar": torch.rand(511, 2),
    "pos_cur": torch.rand(511, 2),
    "pos_tar": torch.rand(511, 2),

    "l1_dense_edge_index_cur": torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  2,
          2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,
          5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,
          6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
          7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,
          9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12,
         12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
         13, 13, 13, 13, 13, 13, 13],
        [ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,
          6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10,
         11, 12, 13,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,
          3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,  6,  7,
          8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
         13,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,
          5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,  6,  7,  8,  9,
         10, 11, 12, 13,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,
          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  2,  3,  4,  5,  6,
          7,  8,  9, 10, 11, 12, 13]]),

    "l1_dense_edge_index_tar": torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,
          2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,
          5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,
          6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
          7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
          9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
         11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
         12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,
          4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,
          8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
         12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,
          2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,
          6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
         10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,
          4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,
          8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
         12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]]),
    "l0_to_l1_edge_index_j_cur": torch.tensor([ 45, 372, 267, 268, 271, 277, 286, 295, 394, 395, 396, 413, 479, 476,
        501, 164, 165, 172, 323,   7,  17, 320, 324, 330, 352, 230, 253, 362,
        367, 454, 463, 495, 259, 307, 309, 312, 409, 415, 421, 424, 428, 474,
        478, 504, 190, 328, 336, 347, 436, 442, 486, 404, 423, 429, 337, 338,
        345, 231, 364, 365, 370, 492, 497, 507, 334, 432]),

    "l0_to_l1_edge_index_i_cur": torch.tensor([ 0,  0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  4,  4,  4,
         4,  5,  5,  5,  5,  5,  6,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,
         8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10,
        11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13]),

    "cluster_mask": torch.tensor([ True, False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True]),
    "cluster_centers_index": torch.tensor([ 56,  64, 405, 131, 172, 174, 213, 244, 297, 340, 296, 437, 456, 485]),
    # batch = None
    "num_clusters": torch.tensor(14),
}

ckpt=torch.load("checkpoints/cns_state_dict.pth", "cpu")
if hasattr(ckpt, "net") and isinstance(ckpt["net"], torch.nn.Module):
    model: OVGraphVS = ckpt["net"]
else:
    model = OVGraphVS(2, 2, 128, regress_norm=True).to("cpu")
    model.load_state_dict(ckpt)

# ov_model = openvino.convert_model(model, example_input=example_input)

# openvino.save_model(ov_model, "cns_ov/openvino_model_new_scene.xml")

# example_input["hidden"] = torch.rand(14,128)
# ov_model_old = openvino.convert_model(model, example_input=example_input)
# openvino.save_model(ov_model_old, "cns_ov/openvino_model_old_scene.xml")

example_input["hidden"] = torch.rand(14,128)
example_input["magic_number"] = torch.tensor([1])
ov_model_old = openvino.convert_model(model, example_input=example_input)
openvino.save_model(ov_model_old, "cns_ov/openvino_model.xml")

# core = openvino.Core()
# model_= core.compile_model("cns_ov/openvino_model_new_scene.xml")
# print(model_(example_input))

# example_input["hidden"] = torch.rand(14,128)
# example_input["magic_number"] = torch.tensor([1])
# model = core.compile_model("cns_ov/openvino_model.xml")
# print(model(example_input))