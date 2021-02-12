

with open("train/feats.scp") as f:
    lines = f.read().splitlines()

utt2feat = {}
for l in lines:
    utt2feat[l.split()[0]] = (l.split()[1],"TRA")


with open("val/feats.scp") as f:
    lines = f.read().splitlines()

import random

for l in lines:
    k = random.randint(0, 1)
    if k:
        utt2feat[l.split()[0]] = (l.split()[1],"DEV")
    else:
        utt2feat[l.split()[0]] = (l.split()[1],"EVA")


with open("train/utt2lang") as f:
    lines = f.read().splitlines()

utt2lang = {}
for l in lines:
    utt2lang[l.split()[0]] = l.split()[1]

with open("val/utt2lang") as f:
    lines = f.read().splitlines()

for l in lines:
    utt2lang[l.split()[0]] = l.split()[1]

a = list(utt2feat.keys())
random.shuffle(a)

with open("source.csv","w") as f:
    f.write("uttr_id,language,split,feat_path\n")
    for utt in a:
        f.write(utt+","+utt2lang[utt]+","+utt2feat[utt][1]+","+utt2feat[utt][0]+"\n")
