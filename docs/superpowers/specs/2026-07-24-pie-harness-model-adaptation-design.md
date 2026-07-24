# PIE + auto-infer Harness and Model Adaptation Design

## Goal

Turn model adaptation into a resumable, machine-verifiable workflow while
keeping the production runtime independent of Agent tooling.

The system has two owners:

- **PIE owns the Agent control plane**: resource discovery, architecture
  investigation, knowledge retrieval, adaptation decisions, remote iteration,
  checkpoints, and escalation.
- **auto-infer owns the deterministic framework Harness**: checkpoint
  inspection, capability matching, model-package generation, static/runtime
  validation, structured artifacts, and model-package loading.

Neither side may duplicate the other's responsibility.

## Rejected Alternatives

### Put orchestration inside auto-infer

This would couple the runtime repository to web discovery, SSH/container
policy, Agent checkpoints, and framework-specific knowledge. It would make the
stable core larger and less locally understandable.

### Put all adaptation logic in PIE prompts

This would make runtime capability checks, package shape, and validation
semantics implicit text. Prompt drift could produce packages that PIE calls
successful but auto-infer cannot load.

## Boundary

```text
PIE
  gather → investigate → choose strategy → edit candidate package → iterate
                                      │
                                      │ StepEnvelope-compatible JSON
                                      ▼
inferctl
  inspect → capability match → generate → validate → structured artifacts
                                      │
                                      │ model-package.json
                                      ▼
auto-infer stable runtime
  explicit package registration → existing model/backend seams → execution
```

The runtime imports no PIE module. PIE does not import auto-infer internals; it
invokes the public `inferctl` command.

## Stable Artifacts

### Harness envelope

Every command writes one JSON object compatible with PIE's `StepEnvelope`:

```json
{
  "status": "ok",
  "step_id": "inspect-model",
  "created_at": "2026-07-24T00:00:00Z",
  "error_summary": null,
  "artifacts": {"model_manifest": ".../model-manifest.json"},
  "provenance": {
    "framework_commit": "...",
    "config_sha256": "...",
    "python": "...",
    "platform": "..."
  },
  "result": {}
}
```

Exit codes are stable: `0=ok`, `2=partial`, `1=failed`.

### Inspection manifest

`model-manifest.json` is generated from checkpoint facts, not Agent guesses. It
records:

- source architecture and config fingerprint;
- attention family and dimensions;
- dense/MoE/MTP/sliding-window features;
- weight-name evidence when an index or safetensors headers are available;
- cache geometry;
- required and missing runtime capabilities.

### Model package

`model-package.json` is the only runtime adaptation descriptor:

```json
{
  "schema_version": 1,
  "name": "example-model",
  "architectures": ["ExampleForCausalLM"],
  "source": {"config_sha256": "..."},
  "implementation": {
    "template": "gqa-swiglu-v1",
    "entrypoint": "auto_infer.models.qwen2:Qwen2Model"
  },
  "execution": {
    "attention": "gqa",
    "dtype": "bfloat16",
    "quantization": {"enabled": false, "interface": "reserved"}
  },
  "validation": {"static": "pending", "runtime": "pending"}
}
```

A package may use a relative Python entrypoint such as
`./model.py:GeneratedModel` when a model needs code. Generated code stays in
the package directory and never enters the stable runtime by default.

## Capability Matching

The first implementation recognizes only contracts the current runtime can
prove:

1. **GQA/MHA/MQA + SwiGLU + standard HF weight layout**
   - Qwen2-compatible dimensions use `Qwen2Model`.
   - explicit non-derived `head_dim` or Q/K norm evidence uses `Qwen3Model`.
2. **DeepSeek-style MLA + MoE + YaRN fields**
   - uses `DeepseekV2Model`.
3. **MTP**
   - detected from weight names;
   - package records the capability;
   - graph MTP remains subject to the existing runtime depth and attention
     capability gates.
4. **Quantization**
   - BF16 is the generated default;
   - the package retains a quantization policy interface but automatic
     quantization is out of scope.

Missing config fields, incompatible dimensions, unknown attention, or
unverified weight layout produce `partial`, never a runnable package marked
`ok`.

## Harness CLI

```bash
inferctl capabilities
inferctl inspect model MODEL --artifacts DIR
inferctl adapt model MODEL --output PACKAGE_DIR --artifacts DIR
inferctl validate package PACKAGE_DIR --model MODEL --artifacts DIR
```

Commands are non-interactive, emit JSON, use fixed artifact names, capture
provenance, and never mutate the checkpoint directory.

## Runtime Loading

`ModelConfig` accepts an optional model-package path. The composition root
registers it before executor construction. The existing architecture registry
then resolves the checkpoint's `architectures[0]` exactly as it resolves
built-ins.

Registration validates:

- schema version;
- source config fingerprint;
- architecture names;
- entrypoint form;
- duplicate mappings;
- model contract methods.

The scheduler, KV manager, serving layer, graph runners, and attention
backends are unchanged.

## PIE Workflow

PIE gains an `adapt-auto-infer` skill:

1. create/resume its checkpoint;
2. gather and investigate the model using existing PIE skills;
3. run `inferctl inspect model`;
4. if capability status is supported, run deterministic package generation;
5. if partial, edit only the candidate package or escalate a reusable
   capability addition to auto-infer;
6. run `inferctl validate package`;
7. sync the candidate and checkpoint to the NPU environment;
8. run runtime accuracy gates;
9. write a StepEnvelope-compatible final result.

PIE treats `inferctl` output as authoritative. It must not rewrite an `ok`
result based on subjective interpretation.

## Error Handling

- Invalid checkpoint/config: `failed`, exit 1.
- Recognized model with missing evidence/capability: `partial`, exit 2.
- Invalid package or fingerprint drift: `failed`, exit 1.
- Runtime import/load failure: validation failure; no fallback to a built-in
  architecture with a different name.
- Existing built-in architecture: package registration is idempotent only when
  it resolves to the same class.

## Verification

- host tests for inspection, capability matching, deterministic generation,
  envelope/exit-code behavior, package validation, and registry loading;
- regression tests proving no model branch enters EngineCore or runners;
- PIE repository checks for skill metadata, framework routing, public CLI use,
  checkpoint protocol, and StepEnvelope output;
- NPU smoke using a generated package alias for Qwen3, followed by token
  equality against the built-in path.

## Non-Goals

- generating arbitrary novel attention or multimodal code without Agent work;
- silently modifying stable runtime files from PIE;
- automatically enabling quantization;
- replacing PIE's existing vllm-ascend adaptation flow;
- duplicating benchmark/profile implementation already owned by auto-infer.

