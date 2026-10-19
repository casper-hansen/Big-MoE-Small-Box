"""The first run downloads the 125 GB checkpoint and builds the runtime; later runs start in about a minute."""
from bigmoe import AutoBox

box = AutoBox.from_pretrained("glm-5.3-flash", context=131072)

with box:  # stops the server at the end only if it started it here
    print(box.chat("In two sentences: why can a 320B-parameter MoE model run on one 24 GB GPU?"))

    for piece in box.chat("Write a haiku about NVMe drives.", stream=True, thinking=True):
        print(piece, end="", flush=True)
    print()

    print("OpenAI-compatible endpoint:", box.url)
