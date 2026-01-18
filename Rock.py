import numpy as np
from sklearn.linear_model import LogisticRegression
#empty arr 

X = []
y = []
model = LogisticRegression(multi_class="multinomial", max_iter=200)

def player(prev_play):
    global X, y, model

    if prev_play == "":
        X.clear()
        y.clear()
        return "R"

    if len(y) > 0:
        y.append(prev_play)

    if len(X) >= 5:
        model.fit(X, y)
        prediction = model.predict([X[-1]])[0]
    else:
        prediction = np.random.choice(["R", "P", "S"])

    X.append([{"R":0,"P":1,"S":2}[prev_play]])
  #both of R and P and S expressed as symbols 

    counter = {"R": "P", "P": "S", "S": "R"}
    return counter[prediction]
