import pickle

with open("data/adj_METR-LA.pkl", "rb") as f:
    obj = pickle.load(f, encoding="latin1")

print(type(obj))
if isinstance(obj, tuple):
    print("Tuple length:", len(obj))
elif isinstance(obj, list):
    print("List length:", len(obj))
    print("Element 0 type:", type(obj[0]))
    print("Element 1 type:", type(obj[1]))
    print("Element 2 type:", type(obj[2]))
else:
    print(obj)
