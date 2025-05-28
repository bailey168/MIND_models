# Read paths_HCP-MMP.loc into a set for fast lookup
with open('/Users/baileyng/MIND_models/paths_HCP-MMP.loc', 'r') as f:
    hcp_mmp_paths = set(line.strip() for line in f if line.strip())

# Open the output file for writing
with open('/Users/baileyng/MIND_models/paths_aparc.loc', 'r') as aparc, open('/Users/baileyng/MIND_models/paths_merged.loc', 'w') as out:
    for line in aparc:
        path = line.rstrip('\n')
        if path in hcp_mmp_paths:
            out.write(f"{path} HCP-MMP\n")
        else:
            out.write(f"{path}\n")