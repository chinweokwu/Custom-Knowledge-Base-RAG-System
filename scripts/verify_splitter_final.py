import os
import sys
sys.path.append(os.getcwd())
try:
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
"""
    chunks = structural_splitter.split_text(test_text)
    print(f'SUCCESS: Identified {len(chunks)} chunks.')
    for i, c in enumerate(chunks):
        print(f'\n--- Chunk {i+1} ---')
        print(c.strip())
except Exception as e:
    print(f'ERROR: {e}')
