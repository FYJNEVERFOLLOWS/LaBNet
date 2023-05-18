import os 
import json

libri_spkinfo_path = "/CDShare3/LibriSpeech/SPEAKERS.TXT"
spkid2gender_path = "/Work21/2021/fuyanjie/pycode/LaBNet/data/libri_spkid2gender.json"
lines = open(libri_spkinfo_path, "r").readlines()
kv = dict()

for idx, line in enumerate(lines):
    if idx < 12:
        continue
    cols = line.split('|')
    print(f'{cols} ')
    spkid = cols[0].strip()
    gender = cols[1].strip()
    kv[spkid] = gender
    print(f'{spkid}: {kv[spkid]} ')

f = open(spkid2gender_path, 'w')
f.write(json.dumps(kv))
f.close()
