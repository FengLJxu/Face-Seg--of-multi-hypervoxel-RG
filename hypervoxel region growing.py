import open3d as o3d
import numpy as np
import time

start_time = time.time()
# 加载点云
pcd = o3d.io.read_point_cloud("combined1_cathedral1 p (7 48).txt", format='xyzrgb')

# 构建KDTree
pcd.estimate_normals()#估计点云的法线向量
pcd.orient_normals_towards_camera_location()
pcd.normalize_normals()#归一化法线向量
kdtree = o3d.geometry.KDTreeFlann(pcd)

# 定义计算曲率的函数
def compute_curvature(pcd):
    curvature = np.abs(np.asarray(pcd.normals)[:, 2])#曲率是根据法线的Z分量来计算的
    return curvature

# 定义区域生长分割函数
def region_growing_segmentation(pcd, kdtree, distance_threshold, angle_threshold, min_points):
    # 计算曲率
    curvature = compute_curvature(pcd)
    indices = np.argsort(curvature)#排序
    seeds = []
    segments = []
    visited = set()#保存已经访问过的点的索引
    for i in indices:
        if i in visited:
            continue
        seed = i
        seeds.append(seed)
        segment = [seed]#对于每个未访问的点，将其设为种子点，并将其添加到种子点集合和分割结果中
        visited.add(seed)
        #对于每个索引i，如果i已经被访问过，则继续下一个循环。否则，将i设为种子点，并将其添加到seeds和segment中，并将其索引添加到visited中
        while seeds:
            seed = seeds.pop(0)
            seed_normal = np.asarray(pcd.normals)[seed]
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[seed], 30)#30
            for j in idx[1:]:
                if j in visited:
                    continue
            #使用一个while循环，从seeds中取出一个种子点。获取该种子点的法向量，并搜索与该点最近的30个点。并使用KD树搜索与该点最近的30个点的索引。对于每个索引j，如果j已经被访问过，则继续下一个循环
                neighbor_normal = np.asarray(pcd.normals)[j]
                dot_product = np.dot(seed_normal, neighbor_normal)#种子点法向量与邻居点法向量的点积
                # 确保点积在 [-1, 1] 范围内 浮点数精度问题导致的
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)#计算点积的角度，并将其转换为度数。
                if angle * 180 / np.pi < angle_threshold:
                    distance = np.linalg.norm(pcd.points[seed] - pcd.points[j])
                    if distance < distance_threshold:
                        seeds.append(j)
                        segment.append(j)
                        visited.add(j)
                #如果角度小于角度阈值，则计算种子点与邻居点之间的距离。如果距离小于距离阈值，则将邻居点j添加到seeds和segment中，并将其索引添加到visited中
        if len(segment) >= min_points:
            segments.append(segment)
    return segments
#如果segment中的点数大于等于最小点数，则将segment添加到segments中

# 参数设置
distance_threshold = 0.2  # 尝试减小距离阈值0.2
angle_threshold = 3  # 尝试增加角度阈值，允许更大的角度差异3
min_points = 20  # 可能需要增加最小点数，取决于点云的密度20

# 执行生长分割
segments = region_growing_segmentation(pcd, kdtree, distance_threshold, angle_threshold, min_points)

# 可视化和保存结果
pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))  # 初始化所有点为黑色
# 为每个检测到的平面分配不同的颜色
for i, segment in enumerate(segments):
  color = np.random.rand(3)# 随机生成颜色
  for j in segment:
    pcd.colors[j] = color

# 提取 xyzrgb 数据
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 将颜色转换为 0-255 的整数
colors = (colors * 255).astype(int)

# 组合为 xyzrgb 格式
xyzrgb = np.hstack((points, colors))

# 保存为 .txt 文件，保留原始精度
np.savetxt("segmented_combined1_cathedral1 p (7 48).txt", xyzrgb, fmt='%.8f %.8f %.8f %d %d %d', delimiter=' ')

print("已将分割后的点云保存为 txt 文件")
end_time = time.time()
print(f"运行时间: {end_time - start_time} 秒")