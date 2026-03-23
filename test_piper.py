from piper import PiperVoice
import sys

model = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\kodep\voice-lab\models\en_US-lessac-medium.onnx"
v = PiperVoice.load(model)
chunks = list(v.synthesize("Hello world"))
print(f"chunks: {len(chunks)}")
c = chunks[0]
print(f"type: {type(c)}")
underscore = "_"
print(f"fields: {[a for a in dir(c) if not a.startswith(underscore)]}")
print(f"audio type: {type(c.audio)}, len: {len(c.audio)}")
if hasattr(c.audio, "dtype"):
    print(f"dtype: {c.audio.dtype}, nbytes: {c.audio.nbytes}")
