import numpy as np
import time

a = np.zeros(([3]*3))
list_b = []
b = np.random.randint(0,3,(a.ndim,5))
for i in range(a.ndim):
    list_b.append(b[i])
print("b：", b)
print("list_b: ", list_b)
print("tuple(b)",tuple(b))
print("tuple(list_b)", tuple(list_b))

def ravel_index(b, shp):
    return np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1],[1])).dot(b)

start = time.time()
for _ in range(1000):
    np.take(a,ravel_index(b, a.shape))
end = time.time()
print(start, end)
print("Time/loop(ms): ", (end-start)/10**6)
print(a)

start = time.time()
for _ in range(1000):
    a[tuple(b)] += 2
end = time.time()
print(start, end)
print("Time/loop(ms): ", (end-start)/10**6)
print(a)

# start = time.time()
# for _ in range(1000):
#     for i in range(b.shape[1]):
#         a[b[0][i],b[1][i],b[2][i]]
# end = time.time()
# print(start, end)
# print("Time/loop(ms): ", (end-start)/10**6)

c = np.arange(30).reshape((3,10))
d = np.array([[0,1,2]]).T
print(c,d,c+d)