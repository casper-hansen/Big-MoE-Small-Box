"""/reset clears the history, Ctrl-D quits."""
from bigmoe import AutoBox

box = AutoBox("glm-5.3-flash")
if not box.running:
    box.start()
history = []
while True:
    try:
        q = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q == "/reset":
        history = []
        continue
    if not q:
        continue
    history.append({"role": "user", "content": q})
    answer = ""
    for piece in box.chat(history, stream=True):
        answer += piece
        print(piece, end="", flush=True)
    print()
    history.append({"role": "assistant", "content": answer})
