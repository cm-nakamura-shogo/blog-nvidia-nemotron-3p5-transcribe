import base64, json, sys, time, urllib.request
audio = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
payload = {"model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
 "messages": [{"role": "user", "content": [{"type": "text", "text": "Transcribe the following audio."},
   {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}}]}],
 "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False}
for i in range(2):
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=900)
        print(f"request{i+1} OK {time.time()-t:.2f}s:", json.load(r)["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"request{i+1} FAIL {time.time()-t:.1f}s:", e); break
