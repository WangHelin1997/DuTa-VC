# see: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/480
import re
# lexicon = open("/data/lmorove1/hwang258/Speech-Backbones/DiffVC/demos/librispeech-lexicon.txt").readlines()
# sp = re.compile("\s+")
# with open("modified_librispeech-lexicon.txt", "w") as f:
#     for line in lexicon:
#         word, *phonemes = sp.split(line.strip())
#         phonemes = " ".join(phonemes)
#         f.write(f"{word}\t{phonemes}\n")

lexicon = open("./cmu_dictionary.txt", encoding = "ISO-8859-1").readlines()
sp = re.compile("\s+")
with open("./modified_cmu_dictionary.txt", "w") as f:
    for line in lexicon:
        word, *phonemes = sp.split(line.strip())
        phonemes = " ".join(phonemes)
        f.write(f"{word}\t{phonemes}\n")