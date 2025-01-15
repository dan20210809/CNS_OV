import openvino as ov
import numpy as np
import torch
import pickle
import sys


print("OpenVINO Version: {}".format(ov.__version__))
print("Test Infer on: {}".format(sys.argv[1]))

core = ov.Core()

model = "cns_ov/openvino_model.xml"
dev = sys.argv[1]
input_file = "ov_input_data.txt"

compiled_model = core.compile_model(model=model, device_name=dev)

file = open(input_file, 'rb')
example_input = pickle.load(file)
print(type(example_input))
file.close()


result_infer = compiled_model(example_input)

print(result_infer)







