"""Smoke-test: run every editing method on a real model with 1 edit request."""

import json
import sys
import time
import traceback

sys.path.insert(0, "src")

LOG_PATH = "/home/liumingxuan/.cursor/debug-4f095e.log"

def log(hypothesis_id, location, message, data=None):
    import os
    entry = json.dumps({
        "sessionId": "4f095e",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }, ensure_ascii=False)
    with open(LOG_PATH, "a") as f:
        f.write(entry + "\n")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_PATH = "/home/liumingxuan/model/Qwen_2.5-7B-Instruct"

    log("setup", "test:load_model", "Loading model and tokenizer", {"model": MODEL_PATH})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    log("setup", "test:model_loaded", "Model loaded", {
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
        "model_type": model.config.model_type,
    })

    # Check layer names
    layer_names = [n for n, _ in model.named_modules() if "layers.0." in n]
    log("H1", "test:layer_names", "Layer 0 module names", {"names": layer_names[:20]})

    param_names = [n for n, _ in model.named_parameters() if "layers.0." in n]
    log("H1", "test:param_names", "Layer 0 param names", {"names": param_names[:20]})

    from trainer.edit.base import EditRequest
    from data.editing import ZSREDataset

    ds = ZSREDataset()
    edit_requests = ds.to_edit_requests(limit=1)
    req = edit_requests[0]
    log("setup", "test:request", "Edit request", {
        "prompt": req.prompt[:80],
        "subject": req.subject,
        "target_new": req.target_new,
    })

    from transformers import TrainingArguments
    import tempfile, os

    results = {}

    EDITORS = [
        ("ROMEEditor", "trainer.edit.rome", "ROMEEditor", {
            "layers": [5],
            "v_num_grad_steps": 3,
        }),
        ("MEMITEditor", "trainer.edit.memit", "MEMITEditor", {
            "layers": [4, 5],
            "v_num_grad_steps": 3,
        }),
        ("MENDEditor", "trainer.edit.mend", "MENDEditor", {
            "layers": [5],
            "n_hidden": 64,
            "rank": 64,
        }),
        ("AlphaEditEditor", "trainer.edit.alphaedit", "AlphaEditEditor", {
            "layers": [4, 5],
            "v_num_grad_steps": 3,
            "v_lr": 0.5,
            "v_weight_decay": 0.5,
            "v_loss_layer": 27,
            "clamp_norm_factor": 4.0,
            "nullspace_threshold": 1e-5,
            "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
            "layer_module_tmp": "model.layers.{}",
            "lm_head_module": "lm_head",
            "ln_f_module": "model.norm",
            "stats_dir": "/tmp/edit_stats",
            "mom2_dataset": "wikipedia",
            "mom2_n_samples": 100,
        }),
        ("UNKEEditor", "trainer.edit.unke_editor", "UNKEEditor", {
            "layers": [4, 5],
            "v_num_grad_steps": 3,
            "v_lr": 0.5,
            "v_weight_decay": 0.5,
            "v_loss_layer": 27,
            "clamp_norm_factor": 4.0,
            "ft_lr": 1e-5,
            "ft_epochs": 2,
            "weight_decay_factor": 0.1,
            "max_weight_deviation": 0.1,
            "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
            "layer_module_tmp": "model.layers.{}",
            "lm_head_module": "lm_head",
            "ln_f_module": "model.norm",
        }),
        ("GRACEEditor", "trainer.edit.grace", "GRACEEditor", {
            "layers": [5],
            "inner_params": "model.layers.5.mlp.down_proj",
            "edit_lr": 0.1,
            "n_iter": 5,
            "eps": 5e-4,
            "num_pert": 2,
        }),
        ("WISEEditor", "trainer.edit.wise", "WISEEditor", {
            "layers": [5],
            "inner_params": "model.layers.5.mlp.down_proj",
            "edit_lr": 0.1,
            "n_iter": 5,
            "merge_strategy": "slerp",
            "merge_freq": 5,
        }),
        ("IKEEditor", "trainer.edit.ike", "IKEEditor", {
            "sentence_model_name": "all-MiniLM-L6-v2",
            "k": 2,
            "use_icl_examples": False,
        }),
        ("SERACEditor", "trainer.edit.serac", "SERACEditor", {
            "archive": None,
            "edit_lr": 1e-4,
            "num_edit_steps": 3,
        }),
        ("MALMENEditor", "trainer.edit.malmen", "MALMENEditor", {
            "layers": [5],
            "archive": None,
            "edit_lr": 1e-4,
            "n_hidden": 64,
            "rank": 64,
        }),
        ("InstructEditEditor", "trainer.edit.instructedit", "InstructEditEditor", {
            "layers": [5],
            "n_hidden": 64,
            "rank": 64,
        }),
        ("AnyEditEditor", "trainer.edit.anyedit", "AnyEditEditor", {
            "layers": [4, 5],
            "strategy": "memit",
            "v_num_grad_steps": 3,
            "v_lr": 0.5,
            "v_weight_decay": 0.5,
            "v_loss_layer": 27,
            "clamp_norm_factor": 4.0,
            "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
            "layer_module_tmp": "model.layers.{}",
            "lm_head_module": "lm_head",
            "ln_f_module": "model.norm",
        }),
    ]

    for editor_name, module_path, class_name, method_args in EDITORS:
        log("test", f"test:{editor_name}", f"--- Testing {editor_name} ---")

        # Reload model weights to avoid contamination between methods
        if editor_name not in ("ROMEEditor",):
            pass  # We'll use the same model for speed; edits are small

        try:
            mod = __import__(module_path, fromlist=[class_name])
            EditorClass = getattr(mod, class_name)

            with tempfile.TemporaryDirectory() as tmp_dir:
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    report_to=[],
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    logging_strategy="no",
                    save_strategy="no",
                    eval_strategy="no",
                    disable_tqdm=True,
                    use_cpu=False,
                )

                log("test", f"test:{editor_name}:init", f"Instantiating {editor_name}", {"method_args_keys": list(method_args.keys())})

                editor = EditorClass(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    **method_args,
                )

                log("test", f"test:{editor_name}:edit", f"Calling edit() on {editor_name}")
                t0 = time.time()
                result = editor.edit(req)
                elapsed = time.time() - t0

                log("test", f"test:{editor_name}:result", f"{editor_name} completed", {
                    "result": str(result)[:500],
                    "elapsed_s": round(elapsed, 2),
                    "success": result.get("success", "N/A") if isinstance(result, dict) else "N/A",
                })
                results[editor_name] = {"status": "OK", "elapsed": round(elapsed, 2), "result": str(result)[:200]}
                print(f"  [PASS] {editor_name} ({elapsed:.1f}s)")

        except Exception as e:
            tb = traceback.format_exc()
            log("test", f"test:{editor_name}:error", f"{editor_name} FAILED", {
                "error": str(e)[:500],
                "traceback": tb[-1000:],
            })
            results[editor_name] = {"status": "FAIL", "error": str(e)[:300]}
            print(f"  [FAIL] {editor_name}: {str(e)[:120]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v["status"] == "OK")
    failed = sum(1 for v in results.values() if v["status"] == "FAIL")
    print(f"Passed: {passed}/{len(results)}, Failed: {failed}/{len(results)}")
    for name, res in results.items():
        status = "PASS" if res["status"] == "OK" else "FAIL"
        detail = res.get("elapsed", res.get("error", ""))
        print(f"  [{status}] {name}: {detail}")

    log("summary", "test:summary", "Test summary", {
        "passed": passed,
        "failed": failed,
        "details": {k: v["status"] for k, v in results.items()},
    })


if __name__ == "__main__":
    main()
