import torch_npu
ops=[o for o in dir(torch_npu) if "weight_quant" in o.lower() or "w4" in o.lower() or "int4" in o.lower() or ("quant" in o.lower() and "matmul" in o.lower())]
print("candidate W4/weight-quant ops:", ops)
# try npu_weight_quant_batchmatmul (A8W4 / A16W4) if present
dev="npu:0"
if hasattr(torch_npu,"npu_weight_quant_batchmatmul"):
    print("has npu_weight_quant_batchmatmul")
