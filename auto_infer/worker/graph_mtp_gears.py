"""Fixed-address buffer owners for graph MTP capture gears."""
import torch


class TargetGear:
    def __init__(self, gear, max_blocks, hidden, device, dtype, geometry):
        self.g = gear
        self.geometry = geometry
        rows = gear * geometry.query_width
        self.tid = torch.zeros(rows, dtype=torch.long, device=device)
        self.ppos = torch.zeros(rows, dtype=torch.long, device=device)
        self.pslot = torch.zeros(rows, dtype=torch.int32, device=device)
        self.bt = torch.zeros(gear, max_blocks, dtype=torch.int32, device=device)
        self.drafts = torch.zeros(
            gear, geometry.draft_depth, dtype=torch.long, device=device)
        self.active_mask = torch.zeros(gear, dtype=torch.int32, device=device)
        self.ep_active_token_mask = torch.zeros(
            rows, dtype=torch.bool, device=device)
        self.p_buf = torch.zeros(
            gear, geometry.query_width, dtype=torch.long, device=device)
        self.na_buf = torch.zeros(gear, dtype=torch.long, device=device)
        result_columns = geometry.query_width + 1 + geometry.draft_depth
        self.result_buf = torch.zeros(
            gear, result_columns, dtype=torch.long, device=device)
        self.compact_hidden = torch.zeros(
            rows, hidden, dtype=dtype, device=device)
        self.compact_tokens = torch.zeros(rows, dtype=torch.long, device=device)
        self.compact_positions = torch.zeros(
            rows, dtype=torch.long, device=device)
        self.compact_slots = torch.zeros(
            rows, dtype=torch.int32, device=device)
        self.scratch_positions = torch.arange(
            rows, dtype=torch.long, device=device)
        self.scratch_slots = torch.zeros(
            rows, dtype=torch.int32, device=device)
        self.cu = [geometry.query_width * (index + 1)
                   for index in range(gear)]
        self.graph = None
        self.reg = []
        self.pipeline = None
        self.stager = None


class DrafterGear:
    def __init__(self, key, target, max_blocks, device):
        token_gear, request_gear = key
        self.key = key
        self.token_gear = token_gear
        self.request_gear = request_gear
        self.target = target
        self.block_table = torch.zeros(
            request_gear + 1, max_blocks, dtype=torch.int32, device=device)
        self.sample_rows = torch.zeros(
            request_gear, dtype=torch.long, device=device)
        self.draft_buf = torch.zeros(
            request_gear, target.geometry.draft_depth,
            dtype=torch.long, device=device)
        self.state_buf = torch.zeros(
            request_gear, target.compact_hidden.shape[1],
            dtype=target.compact_hidden.dtype, device=device)
        self.graph = None
        self.reg = []
        self.pipeline = None
        self.stager = None


class ContinuationGear:
    def __init__(self, request_gear, max_blocks, hidden, device, dtype, depth):
        steps = depth - 1
        self.hidden_in = torch.zeros(
            request_gear, hidden, dtype=dtype, device=device)
        self.token_in = torch.zeros(
            request_gear, dtype=torch.long, device=device)
        self.draft_out = torch.zeros(
            request_gear, steps, dtype=torch.long, device=device)
        self.positions = torch.zeros(
            steps, request_gear, dtype=torch.long, device=device)
        self.slots = torch.zeros(
            steps, request_gear, dtype=torch.int32, device=device)
        self.block_table = torch.zeros(
            request_gear, max_blocks, dtype=torch.int32, device=device)
        self.cu = list(range(1, request_gear + 1))
        self.graph = None
        self.reg = []
        self.pipeline = None
        self.stager = None
