import sys

from bigmoe import AutoBox

box = AutoBox("glm-5.3-flash")
for step in box.prepare(dry_run=True):
    print(f"{step['step']:8s} {step['status']:8s} {step['title']}\n{'':17s}{step['detail']}")
if "--dry-run" not in sys.argv:
    for step in box.prepare():
        print(f"{step['step']:8s} {step['status']}")
