import numpy as np
import torch
import pickle
import sys
import time
from cns.models.graph_vs import GraphVS


print("Test Infer on: CPU")
print("Infer Loop count: {}".format(sys.argv[1]))


model = "cns_ov/openvino_model.xml"
loop_cnt = int(sys.argv[1])

dump_file_data = 'pt_input_data.txt'
dump_file_hidden = 'pt_input_hidden.txt'

device='cpu'
net = GraphVS(2, 2, 128, regress_norm=True).to(device)
ckpt_path="checkpoints/cns_state_dict.pth"
ckpt = torch.load(ckpt_path, device)
net.load_state_dict(ckpt)

net.eval()

file = open(dump_file_data, 'rb')
input_data = pickle.load(file)
#print(type(example_input))
file.close()

file = open(dump_file_hidden, 'rb')
input_hidden = pickle.load(file)
#print(type(example_input))
file.close()

start_tm = time.time()
for i in range(0, loop_cnt):
    raw_pred = net(input_data, input_hidden)
duration_tm = time.time() - start_tm

print("Infer done!! latency = {:.3f}".format(duration_tm*1000/loop_cnt))

sys.exit()
#print(result_infer)







