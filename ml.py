from pickle import load

def predict(sample:list):
    with open('./model.pcl', 'rb') as fid:
        model2 = load(fid)
    res = model2.predict_proba(sample)
    return res[:, 1]

sample = [[-1.4331398, 0.95133062, 1.76721911, -0.55134134, -0.26759586, 1.04375945, 0, 0, 0, 0, 1, 0, 0, 1, 0]]

print(sample[0][1])
print(predict(sample))
s = [[0]*15]
print(s)
print(predict(s))