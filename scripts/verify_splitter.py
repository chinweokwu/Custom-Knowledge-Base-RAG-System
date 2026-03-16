import os
import sys
sys.path.append(os.getcwd())
from app.services.loaders import structural_splitter

test_text = """
ICTSM Dev Guide
1. How to query records:
To find a ticket, use the following TQL:
var query = `select * from tt_troubleticket where alarmname is $alarmname`;
Do not forget the semicolon.

* Important Steps:
    - Step 1: Open Terminal
    - Step 2: Run Query
    - Step 3: Check Results
    - Step 4: Close Session

| ID | Status | Remark |
|----|--------|--------|
| 1  | Open   | New    |
| 2  | Closed | Fixed  |
"""

print("Running StructuralSplitter Verification...")
chunks = structural_splitter.split_text(test_text)
print(f"Total chunks identified: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.strip())
