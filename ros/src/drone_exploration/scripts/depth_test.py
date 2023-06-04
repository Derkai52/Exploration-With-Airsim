import math
import airsim
import numpy as np

import matplotlib.pyplot as plt


# Width = 240
# Height = 160
# CameraFOV = 90
# Fx = Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
# Cx = Width / 2
# Cy = Height / 2
# Colour = (0, 255, 0)
# RGB = "%d %d %d" % Colour # Colour for points
# Colour = (0, 255, 0)
# RGB = "%d %d %d" % Colour # Colour for points

def depthConversion(PointDepth, f):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = float(H) / 2 - 1
    j_c = float(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
    return PlaneDepth

# def depth_dedistortion(pointDepth, f):
#     #print("depth_dedistortion",time.time())
#     H = pointDepth.shape[0]
#     W = pointDepth.shape[1]
#     i_c = float(H) / 2 - 1
#     j_c = float(W) / 2 - 1
#     columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
#     DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
#     PlaneDepth = pointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
#     #np.savetxt("../data/dedistortion.txt",PlaneDepth,fmt='%.2e')
#     return PlaneDepth

def generatepointcloud(depth):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # z = 1000 * np.where(valid, depth / 256.0, np.nan)
    # x = np.where(valid, z * (c - Cx) / Fx, 0)
    # y = np.where(valid, z * (r - Cy) / Fy, 0)
    z = np.where(valid, depth, np.nan)
    x = (np.multiply(z,(cx-c))/fx).ravel()
    y = (np.multiply(z,(cy-r))/fy).ravel()
    z = z.ravel()
    return np.dstack((x, y, z))[0].T

def savepointcloud(image, filename):
    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if math.isinf(pt[0]) or math.isnan(pt[0]) or pt[0] > 10000 or pt[1] > 10000 or pt[2] > 10000:
                None
            else:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2] - 1, RGB))
    f.close()

client = airsim.CarClient()
client.confirmConnection()

fx = 120.0
fy = 120.0
cx = 120.0  #camera.width/2
cy = 80.0  

responses = client.simGetImages(
    [airsim.ImageRequest('front_center_custom', airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)],vehicle_name = 'Drone_1')
response = responses[0]
img1d = np.array(response.image_data_float, dtype=float)
img1d[img1d > 255] = 255
img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
img2d_converted = depthConversion(img2d, fx)
#pcl = generatepointcloud(img2d)
pcl = generatepointcloud(img2d_converted)
#savepointcloud(pcl, 'pcl.asc')
print(pcl.shape)

fig = plt.figure(figsize=(12, 10),dpi=80)
ax = fig.add_subplot(111, projection='3d')
X = pcl[0]
Y = pcl[1]
Z = pcl[2]
ax.scatter(X, Y, Z,s= 10,edgecolor="blue",marker=".")
plt.show()

