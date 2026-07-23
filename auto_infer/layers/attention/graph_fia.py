"""Shared ACL graph lifecycle for dynamically updated FIA-v2 calls."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _FiaRegistration:
    task: object
    query: object
    key: object
    value: object
    output: object
    lse: object
    block_size: int


class GraphFiaLifecycle:
    """Capture and update policy shared by graph GQA and MLA backends."""

    def _init_graph_fia(self) -> None:
        self.capturing = False
        self.reg: list[_FiaRegistration] = []

    def _fia_call(self, query, key, value, output, lse, ctx, block_size):
        import torch_npu

        return torch_npu.npu_fused_infer_attention_score_v2.out(
            query,
            key,
            value,
            num_query_heads=self._fia_query_heads,
            num_key_value_heads=self._fia_kv_heads,
            input_layout="TND",
            softmax_scale=self._fia_scale,
            block_table=ctx.block_table,
            block_size=block_size,
            sparse_mode=3,
            atten_mask=ctx.attn_mask,
            actual_seq_qlen=ctx.cu_seqlens_q,
            actual_seq_kvlen=ctx.seqlens_kv,
            out=[output, lse],
        )

    def _run_graph_fia(
        self, query, key, value, output, lse, ctx, block_size
    ) -> None:
        if self.capturing:
            import torch

            from auto_infer.graph_tasks import capture_graph_task

            stream = torch.npu.current_stream()
            task = capture_graph_task(
                stream,
                lambda: self._fia_call(
                    query, key, value, output, lse, ctx, block_size),
            )
            self.reg.append(_FiaRegistration(
                task, query, key, value, output, lse, block_size))
            return
        self._fia_call(query, key, value, output, lse, ctx, block_size)

    def begin_capture(self) -> None:
        self.capturing = True
        self.reg = []

    def end_capture(self) -> None:
        self.capturing = False

    def _update_registration(self, registration, ctx, stream) -> None:
        from auto_infer.graph_tasks import update_graph_task

        update_graph_task(
            registration.task,
            stream,
            lambda: self._fia_call(
                registration.query,
                registration.key,
                registration.value,
                registration.output,
                registration.lse,
                ctx,
                registration.block_size,
            ),
        )

    def update(self, ctx, stream=None) -> None:
        self.update_registrations(self.reg, ctx, stream)

    def update_registrations(self, registrations, ctx, stream=None) -> None:
        import torch

        stream = stream or torch.npu.current_stream()
        for registration in registrations:
            self._update_registration(registration, ctx, stream)

    def update_many(self, contexts, stream=None) -> None:
        if len(contexts) != len(self.reg):
            raise ValueError("graph-task contexts must match registrations")
        import torch

        stream = stream or torch.npu.current_stream()
        for ctx, registration in zip(contexts, self.reg):
            self._update_registration(registration, ctx, stream)
