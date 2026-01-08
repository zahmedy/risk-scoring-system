

def gini(y) -> float:
    if len(y) == 0: 
        return 0.0

    counts = {}

    for label in y:
        counts[label] = counts.get(label, 0) + 1
    
    impurity = 1.0
    n = len(y)

    for count in counts.values():
        p = count / n
        impurity -= p ** 2

    return impurity

def gini_gain(y_parent, y_left, y_right) -> float:
    n = len(y_parent)
    if n == 0:
        return 0.0

    if len(y_left) == 0 or len(y_right) == 0: 
        return 0.0

    impurity_parent = gini(y_parent)
    impurity_y_left = gini(y_left)
    impurity_y_right = gini(y_right)
    impurity_child = (len(y_left)/n)*impurity_y_left + (len(y_right)/n)*impurity_y_right

    return impurity_parent - impurity_child