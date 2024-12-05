
with open("test.log", "r") as inp:
    lines = inp.readlines()

info = []
for line in lines:
    token, vals = line.split(":")
    a, b = [float(x.strip()) for x in vals.split(",")]
    info.append((token, a, b))

info = sorted(info)
for x, y in zip(info, info[1:]):
    if x[0] == y[0]:
        print("check", x, y)
        assert abs(x[1] - y[1]) < 1e-8
        assert abs(x[2] - y[2]) < 1e-8