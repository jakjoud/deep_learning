import numpy as np

def merge(pref , max):
    results = []
    for i in range(max):
        f = open("data/"+pref+str(i)+'.csv','r')
        lines = f.readlines()
        vals = []
        for l in lines:
            if len(l) == 0:
                continue
            sv = l.split('.')[0]
            vals.append(int(sv))
        results.append(vals)
    results = np.asarray(results)
    np.savetxt("data/raw_results.csv", results, delimiter=",")
    vals = []
    for i in range(results.shape[1]):
        a = results[:,i]
        counts = np.bincount(a)
        val = np.argmax(counts)
        vals.append(val)
    vals = np.asanyarray(vals)
    np.savetxt("data/final.csv", vals, delimiter=",")

merge('mnist_result', 5)