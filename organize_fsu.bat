# Final Professional Structure for FSULLM Archive
# Run these commands in your FSULLM folder:

# 1. Rename the version folders for clarity
move 1FSULLMA v1-baseline
move 2FSULLMBSTREAMNONCUR v2-stream-noncur
move 3FSU_LLM3STREAMCURR v3-stream-curriculum
move 4FSULLMSTREAMCURRMORPHV1ADP v4-adp-protocol
move 5FSULLMSTREAMCURRMORPHV1ADPAURA v5-adp-aura-compiler
move 6FSULLMSTREAMCURRMORPHV1 v6-morph-hybrid
move 7FSULLMSTREAMCURRMORPHV15FSMART v7-fsmart-hybrid

# 2. Create 'src' folders and move .py scripts for each version
mkdir v1-baseline\src; move v1-baseline\*.py v1-baseline\src
mkdir v2-stream-noncur\src; move v2-stream-noncur\*.py v2-stream-noncur\src
mkdir v3-stream-curriculum\src; move v3-stream-curriculum\*.py v3-stream-curriculum\src
mkdir v4-adp-protocol\src; move v4-adp-protocol\*.py v4-adp-protocol\src
mkdir v5-adp-aura-compiler\src; move v5-adp-aura-compiler\*.py v5-adp-aura-compiler\src
mkdir v6-morph-hybrid\src; move v6-morph-hybrid\*.py v6-morph-hybrid\src
mkdir v7-fsmart-hybrid\src; move v7-fsmart-hybrid\*.py v7-fsmart-hybrid\src
