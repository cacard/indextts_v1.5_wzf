import gc
import os
import time
from typing import Optional

import torch
from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto(language="zh_CN")
MODE = 'local'

class MyDemo():
    
    tts: Optional[IndexTTS] = None
    
    @staticmethod
    def load():
        if MyDemo.tts is None:
            print("Initializing TTS model...")
            MyDemo.tts = IndexTTS(model_dir="checkpoints", cfg_path=os.path.join("checkpoints", "config.yaml"),)
            print("TTS model initialized.")
    
    @staticmethod
    def unload():
        if MyDemo.tts is not None:
            print("Unloading TTS model...")
            try:
                if hasattr(MyDemo.tts, "close"):
                    MyDemo.tts.close()
            except Exception as e:
                print(f">> Warning: failed to close TTS model cleanly: {e}")
            MyDemo.tts = None
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            print("TTS model unloaded.")

    @staticmethod
    def gen_single(prompt_wav, 
                text, 
                max_text_tokens_per_sentence=120, 
                sentences_bucket_max_size=4,
                save_path = None,
                ):

        # makesure loaded
        MyDemo.load()
        assert MyDemo.tts is not None

        output_path = save_path if save_path else os.path.join("outputs", f"spk_{int(time.time())}.wav")
        if save_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

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

        MyDemo.tts.infer(prompt_wav, text, output_path, verbose=False,
                            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                            **kwargs)
        print("生成完成")
        return output_path



# ---------------------------------

if __name__ == "__main__":
    prompt = os.path.abspath('./ref_audio/1.wav')
    text = "你好，欢迎使用IndexTTS，这是测试。"
    save_path = os.path.abspath(f'./ref_output/1-{time.time()}.wav')
    MyDemo.gen_single(prompt, text, save_path=save_path)