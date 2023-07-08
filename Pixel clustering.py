from sklearn.cluster import DBSCAN
import cv2
import numpy as np

# 长寿特征点的帧数阈值
frame_threshold = 20

# DBSCAN 参数
epsilon = 10  # 两点之间的最大距离
min_samples = 5  # 一个点的邻居数量

# 加载视频
cap = cv2.VideoCapture('your_video.mp4')

# 读取第一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 角点检测参数
corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
# 光流参数
lk_params = dict(winSize=(200,200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Shi-Tomasi 角点检测
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

# 创建一个 mask
mask = np.zeros_like(prev_frame)

# 创建一个字典来存储特征点出现的帧数
feature_points = {tuple(map(int, point[0])): 1 for point in prevPts}

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade 方法
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)

    # 选择好的点
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]

    # 画出跟踪的点
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 2)
        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)

        # 更新特征点出现的帧数
        point = tuple(map(int, new))
        if point in feature_points:
            feature_points[point] += 1
        else:
            feature_points[point] = 1

    # 更新前一帧和前一点
    prev_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 提取长寿特征点
longevity_points = {point: count for point, count in feature_points.items() if count > frame_threshold}

# 为聚类准备数据
# 注意：在这里，我们将灰度值和位置信息组合在一起
data = np.array([(*point, prev_gray[point]) for point in longevity_points])

# 使用 DBSCAN 进行聚类
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(data)

# 输出聚类结果
for label, point in zip(db.labels_, data):
    print(f'Label: {label}, Point: {point}')
