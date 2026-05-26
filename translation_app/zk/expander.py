try:
    from .zk_input_builder import ExpansionError, expander_main, expand_runtime_input, save_dense_input_json
except ImportError:
    from zk_input_builder import ExpansionError, expander_main, expand_runtime_input, save_dense_input_json

__all__ = [
    "ExpansionError",
    "expand_runtime_input",
    "save_dense_input_json",
    "expander_main",
]


if __name__ == "__main__":
    try:
        raise SystemExit(expander_main())
    except ExpansionError as ex:
        print(f"[expander] error: {ex}")
        raise SystemExit(1)
