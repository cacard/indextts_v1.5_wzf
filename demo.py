import json
import os
import sys
import threading
import time

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir="checkpoints", cfg_path=os.path.join("checkpoints", "config.yaml"),)


def gen_single(prompt, 
               text, 
               infer_mode = None, 
               max_text_tokens_per_sentence=120, 
               sentences_bucket_max_size=4,
               save_path = None,
               ):

    output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    do_sample = True # 是否采样
    top_p = 0.8
    top_k = 30 # 0--100
    temperature = 1.0
    length_penalty = 0 # -2.0 -- 2.0
    num_beams = 3 # 1--10
    repetition_penalty = 10 # 0.1 --20
    max_mel_tokens = 600
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if infer_mode is None:
        output = tts.infer(prompt, text, output_path, verbose=False,
                           max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                           **kwargs)
        print("生成完成")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.rename(output_path, save_path)
            print(f"已保存到：{save_path}")
    else:
        # 批次推理
        output = tts.infer_fast(prompt, text, output_path, verbose=False,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            sentences_bucket_max_size=(sentences_bucket_max_size),
            **kwargs)


# ---------------------------------

if __name__ == "__main__":
    prompt = os.path.abspath('./ref_audio/1.wav')
    text = "你好，欢迎使用IndexTTS，这是测试。"
    save_path = os.path.abspath(f'./ref_output/1-{time.time()}.wav')
    gen_single(prompt, text, save_path=save_path)