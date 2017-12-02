import numpy as np
import skipthoughts
import pickle

print "Loading skip-thoughts model..."
sent_model=skipthoughts.load_model()

# train split
print "Processing train split"
with open("./tacos_train_0_74.txt") as f:
    train_sent=[l.strip().split(":")[1].rstrip("#").split("#") for l in f]
with open("./tacos_train_0_74.txt") as f:
    train_clips=[l.strip().split(":")[0] for l in f]
print len(train_clips)
print len(train_sent)

clip_sentvecs=[]
for k in range(len(train_sent)):
    print str(k)+"/"+str(len(train_sent))
    train_sent_vecs=skipthoughts.encode(sent_model, train_sent[k])
    clip_sentvecs.append((train_clips[k],train_sent_vecs))

pickle.dump(clip_sentvecs,open("./train_clip-sentvec.pkl","w"))


# val test
print "Processing val split"
with open("./tacos_val_75_99.txt") as f:
    val_sent=[l.strip().split(":")[1].rstrip("#").split("#") for l in f]
with open("./tacos_val_75_99.txt") as f:
    val_clips=[l.strip().split(":")[0] for l in f]
print len(val_clips)
print len(val_sent)


clip_sentvecs=[]
for k in range(len(val_clips)):
    print str(k)+"/"+str(len(val_sent))
    val_sent_vecs=skipthoughts.encode(sent_model, val_sent[k])
    clip_sentvecs.append((val_clips[k],val_sent_vecs))
pickle.dump(clip_sentvecs,open("./val_clip-sentvec.pkl","w"))

# test split
print "Processing test split"
with open("./tacos_test_100_126.txt") as f:
    test_sent=[l.strip().split(":")[1].rstrip("#").split("#") for l in f]
with open("./tacos_test_100_126.txt") as f:
    test_clips=[l.strip().split(":")[0] for l in f]
print len(test_clips)
print len(test_sent)


clip_sentvecs=[]
for k in range(len(test_clips)):
    print str(k)+"/"+str(len(test_sent))
    test_sent_vecs=skipthoughts.encode(sent_model, test_sent[k])
    clip_sentvecs.append((test_clips[k],test_sent_vecs))
pickle.dump(clip_sentvecs,open("./test_clip-sentvec.pkl","w"))
