"""Smoke-test: verify all 5 MM editing methods run without errors.

Strategy:
  - Use Qwen2.5-0.5B-Instruct (text-only, small) for GRACE/WISE/MEND/SERAC
    text-fallback path (image=None).  This validates the inheritance chain,
    adapter installation, optimisation loop, and reset logic.
  - Use Qwen2-VL-7B-Instruct (inference-only) for mm_tokenize / mm_forward
    multimodal pipeline.  No backward / edit loop to avoid OOM.
"""

import json
import sys
import time
import traceback

sys.path.insert(0, "src")

def log(hid, loc, msg, data=None):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{hid}] {loc}: {msg}", flush=True)
    if data:
        for k, v in data.items():
            print(f"         {k}={v}", flush=True)


def main():
    import importlib, importlib.util, importlib.machinery, types
    if importlib.util.find_spec("deepspeed") is None:
        ds = types.ModuleType("deepspeed")
        ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
        ds.__version__ = "0.0.0"
        sys.modules["deepspeed"] = ds

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # =====================================================================
    # Part A: text-model tests (GRACE / WISE / MEND / SERAC / IKE)
    # =====================================================================
    TEXT_MODEL = "/home/liumingxuan/model/Qwen_2.5-7B-Instruct"
    log("setup", "partA:start", "Loading text model", {"model": TEXT_MODEL})

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    log("setup", "partA:loaded", "Text model loaded", {
        "dtype": str(next(model.parameters()).dtype),
        "hidden": model.config.hidden_size,
        "layers": model.config.num_hidden_layers,
    })

    from trainer.edit.base import EditRequest
    req = EditRequest(prompt="The capital of France is", subject="France",
                      target_new="Berlin", target_old="Paris")

    # Resolve inner_params
    inner_params = None
    for c in ["model.layers.0.mlp.down_proj", "model.model.layers.0.mlp.down_proj"]:
        try:
            parts = c.split(".")
            m = model
            for p in parts:
                m = getattr(m, p) if not p.isdigit() else m[int(p)]
            inner_params = c
            break
        except (AttributeError, IndexError):
            pass
    log("setup", "partA:inner_params", "inner_params", {"v": inner_params})

    def make_editor(cls_name, mod_path, extra_kw=None):
        """Instantiate an MM editor without full HF Trainer init."""
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        obj = object.__new__(cls)
        obj.processor = None
        obj.model = model
        obj.tokenizer = tokenizer
        obj.layers = [0]
        obj.preserve_memory = False
        obj.edit_history = []
        if extra_kw:
            for k, v in extra_kw.items():
                setattr(obj, k, v)
        return obj

    # --- MM-IKE ---
    log("H1", "partA:ike_start", "MM-IKE text path")
    try:
        ike = make_editor("MMIKEEditor", "trainer.edit.mm_ike", {
            "sentence_model_name": "all-MiniLM-L6-v2", "k": 3,
            "use_icl_examples": False, "knowledge_store": [], "knowledge_sentences": [],
            "knowledge_embeddings": None, "_sentence_model": None,
            "icl_prompt_template": "New Fact: {fact}\nPrompt: {prompt}\n\n",
            "image_description_template": "[Image context] {prompt}",
        })
        r = ike.edit(req)
        log("H1", "partA:ike_ok", "MM-IKE PASS", {"success": r["success"], "count": r["edited_count"]})
    except Exception as e:
        log("H1", "partA:ike_fail", "MM-IKE FAIL", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    # --- MM-GRACE ---
    log("H1", "partA:grace_start", "MM-GRACE text path")
    try:
        grace = make_editor("MMGRACEEditor", "trainer.edit.mm_grace", {
            "inner_params": inner_params, "target_layer": inner_params,
            "edit_lr": 1e-1, "n_iter": 3, "eps": 0.5, "num_pert": 1,
            "val_init": "warm", "val_train": "sgd", "val_reg": 0.0,
            "dropout": 0.0, "replacement": "replace_last",
            "_adapter_installed": False, "_original_layer": None, "_edit_count": 0,
        })
        r = grace.edit(req)
        log("H1", "partA:grace_ok", "MM-GRACE PASS", {"success": r["success"], "count": r["edited_count"]})
        grace.reset_layer()
        log("H1", "partA:grace_reset_ok", "MM-GRACE reset PASS")
    except Exception as e:
        log("H1", "partA:grace_fail", "MM-GRACE FAIL", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    # --- MM-WISE ---
    log("H1", "partA:wise_start", "MM-WISE text path")
    try:
        wise = make_editor("MMWISEEditor", "trainer.edit.mm_wise", {
            "inner_params": inner_params, "target_layer": inner_params,
            "edit_lr": 1e-4, "n_iter": 3, "merge_strategy": "slerp",
            "merge_freq": 5, "save_freq": 1, "mask_ratio": 0.5,
            "norm_constraint": None, "merge_weight": 0.5, "density": 0.5,
            "act_ratio": 1.0, "gamma": 5.0, "alpha": 20.0, "beta": 5.0,
            "_adapter_installed": False, "_original_layer": None, "_edit_history": [],
        })
        r = wise.edit(req)
        log("H1", "partA:wise_ok", "MM-WISE PASS", {"success": r["success"], "count": r["edited_count"]})
        wise.reset_layer()
        log("H1", "partA:wise_reset_ok", "MM-WISE reset PASS")
    except Exception as e:
        log("H1", "partA:wise_fail", "MM-WISE FAIL", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    # --- MM-SERAC ---
    log("H1", "partA:serac_start", "MM-SERAC text path")
    try:
        serac = make_editor("MMSERACEditor", "trainer.edit.mm_serac", {
            "archive": None, "edit_lr": 1e-4, "cedit": 0.1, "cloc": 1.0, "cbase": 1.0,
            "classifier_hidden_size": None, "num_edit_steps": 3,
            "edit_memory": [], "classifier": None, "counterfactual_model": None,
            "_is_initialized": False,
        })
        r = serac.edit(req)
        log("H1", "partA:serac_ok", "MM-SERAC PASS", {"success": r["success"], "count": r["edited_count"]})
    except Exception as e:
        log("H1", "partA:serac_fail", "MM-SERAC FAIL", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    # --- MM-MEND ---
    log("H1", "partA:mend_start", "MM-MEND gradient computation text path")
    try:
        mend = make_editor("MMMENDEditor", "trainer.edit.mm_mend", {
            "edit_lr": 1e-4, "n_hidden": 128, "rank": 64,
            "edit_network": None, "_initialized": False,
        })
        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(True)
        grad = mend._compute_edit_gradient(req)
        log("H1", "partA:mend_ok", "MM-MEND PASS", {"grad_shape": list(grad.shape), "grad_norm": float(grad.norm().item())})
        model.zero_grad()
    except Exception as e:
        log("H1", "partA:mend_fail", "MM-MEND FAIL", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    # Free text model
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    log("setup", "partA:done", "Part A complete, text model freed")

    # =====================================================================
    # Part B: multimodal pipeline test (mm_tokenize + mm_forward, no backward)
    # =====================================================================
    log("setup", "partB:start", "Loading VL model for inference-only test")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        VL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
        vl_processor = AutoProcessor.from_pretrained(VL_MODEL)
        vl_processor.tokenizer.padding_side = "right"
        vl_model = AutoModelForImageTextToText.from_pretrained(
            VL_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        )
        vl_model.eval()
        log("setup", "partB:loaded", "VL model loaded", {"type": type(vl_model).__name__})

        from PIL import Image
        test_img = Image.new("RGB", (64, 64), color=(128, 64, 32))

        from trainer.edit.mm_mixin import MMEditMixin
        mm = MMEditMixin()
        mm.init_mm(vl_processor)
        mm.model = vl_model

        # mm_tokenize with image
        tokens = mm.mm_tokenize("What color?", test_img, "red", device=torch.device("cpu"))
        log("H2", "partB:tokenize_ok", "mm_tokenize PASS", {
            "keys": list(tokens.keys()), "ids_shape": list(tokens["input_ids"].shape),
            "has_pv": "pixel_values" in tokens,
            "answer_toks": int((tokens["labels"] != -100).sum().item()),
        })

        # mm_tokenize_text_only
        tokens_t = mm.mm_tokenize_text_only("Capital of France?", "Paris")
        log("H2", "partB:tokenize_text_ok", "mm_tokenize_text_only PASS", {
            "keys": list(tokens_t.keys()), "ids_shape": list(tokens_t["input_ids"].shape),
        })

        # mm_forward (inference only)
        dev = next(vl_model.parameters()).device
        tok_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
        with torch.no_grad():
            out = mm.mm_forward(vl_model, tok_dev)
        log("H3", "partB:forward_ok", "mm_forward PASS", {
            "has_loss": out.loss is not None, "loss": float(out.loss.item()) if out.loss is not None else None,
            "logits_shape": list(out.logits.shape),
        })

        del vl_model, vl_processor
        torch.cuda.empty_cache()
        gc.collect()
        log("setup", "partB:done", "Part B complete, VL model freed")
    except Exception as e:
        log("setup", "partB:fail", "Part B FAILED", {"error": str(e), "tb": traceback.format_exc()[-600:]})

    log("setup", "test:done", "ALL SMOKE TESTS COMPLETE")
    print("=== SMOKE TEST COMPLETE - check", LOG_PATH, "===")


if __name__ == "__main__":
    main()
