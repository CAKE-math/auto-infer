"""Process-separated serving: API process (tokenize only, no NPU) <-> EngineCore process.
NOTE: spawn re-imports __main__, so all top-level work MUST be guarded."""
def main():
    from transformers import AutoTokenizer
    from auto_infer.serving.ipc import EngineProcess
    path = "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    eng = EngineProcess(path)
    ids = tok("The capital of France is").input_ids
    if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
        ids = [tok.bos_token_id] + ids
    toks = list(eng.generate_stream("r0", ids, 10))
    print(f"IPC chunks={len(toks)} text={tok.decode(toks)!r}")
    eng.close()


if __name__ == "__main__":
    main()
