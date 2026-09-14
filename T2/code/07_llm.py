"""7단계: 로컬 LLM zero-shot(LLM-0)·few-shot(LLM-FS) 분류. GPU 1장 필요.

- 외부 API를 쓰지 않는다. 모델 입력은 식품코드 사전 문자열(식품명·식품군·대표 제품명·제조사)뿐이다.
- 결정적 추론: 답변 첫 위치에서 토큰 "1"~"4"의 로짓 중 최댓값을 예측으로 한다(샘플링 없음).
- 프롬프트와 few-shot 예시는 prompts.py에 고정되어 있다.

사용법
    CUDA_VISIBLE_DEVICES=0 python 07_llm.py [--scope gold|all] [--batch 16]
    모델: 환경변수 T2_LLM_MODELS (쉼표 구분, 기본 Qwen/Qwen2.5-7B-Instruct)
출력  PRED_DIR/pred_LLM-0@<모델>.csv, pred_LLM-FS@<모델>.csv (N_FCODE, pred, p1..p4)
      PRED_DIR/llm_runinfo.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from prompts import FEWSHOT, build_messages


def score(model, tok, texts, digit_ids, batch):
    probs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]  # 마지막 위치만 계산(메모리 절약)
        sub = logits[:, digit_ids].float()
        probs.append(torch.softmax(sub, dim=-1).cpu().numpy())
    return np.vstack(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="gold", choices=["gold", "all"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    C.ensure_dirs()
    torch.manual_seed(C.SEED)

    t = pd.read_csv(C.RESULTS_DIR / "foodcode_list.csv", dtype={"N_FCODE": str, "식품군코드": str},
                    encoding="utf-8-sig")
    if args.scope == "gold":
        codes = pd.read_csv(C.GOLD, dtype={"N_FCODE": str}, encoding="utf-8-sig").N_FCODE
        t_eval = t[t.N_FCODE.isin(codes)]
    else:
        t_eval = t
    gold_codes = set(pd.read_csv(C.RESULTS_DIR / "gold_sample.csv", dtype={"N_FCODE": str},
                                 encoding="utf-8-sig").N_FCODE)
    bad = [c for c, _ in FEWSHOT if c in gold_codes]
    assert not bad, f"few-shot 예시가 정답셋에 포함됨: {bad}"
    by_code = t.set_index("N_FCODE")
    fs_rows = [(by_code.loc[c].to_dict() | {"N_FCODE": c}, lab) for c, lab in FEWSHOT]

    info = {}
    for mid in C.LLM_MODELS:
        short = mid.split("/")[-1]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        digit_ids = [tok.encode(str(k), add_special_tokens=False) for k in (1, 2, 3, 4)]
        assert all(len(x) == 1 for x in digit_ids), digit_ids
        digit_ids = [x[0] for x in digit_ids]
        for mode, fs in (("LLM-0", None), ("LLM-FS", fs_rows)):
            texts = [tok.apply_chat_template(build_messages(r, fs), tokenize=False, add_generation_prompt=True)
                     for r in t_eval.to_dict("records")]
            t1 = time.time()
            p = score(model, tok, texts, digit_ids, args.batch)
            out = pd.DataFrame({"N_FCODE": t_eval.N_FCODE.to_numpy(), "pred": p.argmax(1) + 1})
            for k in range(4):
                out[f"p{k + 1}"] = p[:, k].round(4)
            suffix = "" if args.scope == "gold" else "_all"
            out.to_csv(C.PRED_DIR / f"pred_{mode}@{short}{suffix}.csv", index=False, encoding="utf-8-sig")
            info[f"{mode}@{short}"] = {"n": len(out), "seconds": round(time.time() - t1, 1),
                                       "prompt_tokens_example": len(tok(texts[0]).input_ids),
                                       "pred_dist": out.pred.value_counts().sort_index().to_dict()}
            print(mode, short, info[f"{mode}@{short}"])
        info[short + "_load_seconds"] = round(time.time() - t0, 1)
        del model
        torch.cuda.empty_cache()
    info["gpu"] = torch.cuda.get_device_name(0)
    info["torch"], info["transformers"] = torch.__version__, __import__("transformers").__version__
    (C.PRED_DIR / "llm_runinfo.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
