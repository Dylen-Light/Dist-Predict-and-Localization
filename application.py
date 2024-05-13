import joblib
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import serial
from scipy.optimize import least_squares
import csv

def exec_predict(rssi, rtt, csi):
    print("executing dest prediction with param:{ RSSI:", rssi, ", RTT:", rtt, ", CSI:", csi, "}.")
    raw_data = np.array([[rssi, rtt, csi]])
    raw_data = MinMaxScaler.transform(raw_data.reshape(3, 1))
    raw_data = raw_data.reshape(1, 1, 3)

    dst_predict = model.predict(raw_data)
    dst_predict = dst_predict.reshape(len(dst_predict), 1)
    dst_predict = MinMaxScaler.inverse_transform(dst_predict)
    # dst_predict = dst_predict.reshape(len(dst_predict), 1)
    print(dst_predict[0][0])
    return dst_predict[0][0]


class vec3d:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def error_function(params, anchor_positions, distances):
    num_anchors = len(anchor_positions)
    x, y, z = params

    # Compute distances from estimated position to anchors
    estimated_position = vec3d(x, y, z)
    estimated_distances = np.zeros(num_anchors)
    for i in range(num_anchors):
        estimated_distances[i] = calculate_distance(estimated_position,
                                                    anchor_positions[i])

    # Calculate error as the difference between estimated distances and measured distances
    error = estimated_distances - distances
    return error


def trilateration(anchor_positions, distances):
    # Initial estimate for the target position
    initial_position = np.array([0.0, 0.0, 0.0])

    # Use least squares optimization to minimize the error function
    result = least_squares(error_function,
                           initial_position,
                           args=(anchor_positions, distances))

    # Extract the estimated target position
    estimated_position = result.x
    return estimated_position


# 通过指定标签坐标，计算与各基站的距离
def get_distance(anchor_positions, tag_position):
    num_anchors = len(anchor_positions)
    distances = np.zeros(num_anchors)
    for i in range(num_anchors):
        distances[i] = calculate_distance(tag_position, anchor_positions[i])
    return distances


# if __name__ == '__main__':
#     # 基站坐标
#     anchor_positions = [
#         vec3d(0, 0, 0.5),
#         vec3d(6, 0, 0.5),
#         vec3d(6, 6, 0.5),
#         vec3d(0, 6, 2.5),
#         vec3d(12, 6, 1.5),
#         vec3d(18, 12, 0.5),
#     ]
#
#     # 通过指定标签坐标，计算与各基站的距离
#     # tag_position = vec3d(3, 3, 1)
#     # distances = get_distance(anchor_positions, tag_position)
#     # print(distances)
#
#     # 通过指定距离，计算标签坐标
#     distances = np.array([4.27200187, 4.27200187, 4.27200187, 4.5, 9.5, 17.5])
#     estimated_position = trilateration(anchor_positions, distances)
#     estimated_position = np.round(estimated_position, 3)
#     print("Estimated Position:", estimated_position)
#     # 验证距离以检查答案是否正确
#     for i in range(len(anchor_positions)):
#         print(
#             calculate_distance(
#                 vec3d(estimated_position[0], estimated_position[1],
#                       estimated_position[2]), anchor_positions[i]))

if __name__ == '__main__':
    print("loading model...")
    model = load_model("mlp_model.h5")
    MinMaxScaler = joblib.load('mmScaler')
    ser = serial.Serial('COM12', 115200)
    #设置串口
    distance = 300
    # 设置实际距离，单位mm
    isLoS = 1
    # 设置是否是LoS环境，1为是0为否
    with open('FTM_Predict.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["SN", "Name", "Contribution"])

        if ser.is_open:
            print("serial has opened.")

            # 构造参数数据结构
            anchor_positions = []
            dist = []
            while True:
                data = ser.readline().decode('UTF-8')
                if "predicting" in data:
                    data = "".join(filter(lambda s: s in '0123456789,.', data))
                    print(data)
                    params = data.split(',')
                    # print(params)
                    # params.append(isLoS)
                    params.append(distance)
                    writer.writerow(params)
                    # pred_dist = exec_predict(params[0], params[1], 5)
                    # if int(params[2]) >= len(anchor_positions):
                    #     anchor_positions.append(vec3d(float(params[3]),float(params[4]),0))
                    #     dist.append(pred_dist)
                    # else :
                    #     #一次循环结束
                    #     distances = np.array(dist)
                    #     estimated_position = trilateration(anchor_positions, distances)
                    #     estimated_position = np.round(estimated_position, 3)
                    #     print("Estimated Position:", estimated_position)
                    #     anchor_positions.clear()
                    #     dist.clear()







