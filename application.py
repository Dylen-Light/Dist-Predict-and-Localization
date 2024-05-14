import json
import threading

import joblib
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import serial
from scipy.optimize import least_squares
from flask import Flask
import csv


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,APInformation):
            d = {'__class__': obj.__class__.__name__, '__module__': obj.__module__}
            d.update(obj.__dict__)
            return d
        else:
            return super(MyEncoder, self).default(obj)


class APInformation:
    def __init__(self, apid, pos_x, pos_y, dist, is_valid, is_los):
        self.apid = apid
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.dist = dist
        self.is_valid = is_valid
        self.is_los = is_los


def exec_predict(rssi, rtt, csi):
    # print("executing dest prediction with param:{ RSSI:", rssi, ", RTT:", rtt, ", CSI:", csi, "}.")
    raw_data = np.array([csi])
    raw_data = np.insert(raw_data, 0, rtt)
    raw_data = np.insert(raw_data, 0, rssi)
    # print(raw_data)
    feature_len = 2 + 128
    raw_data = MinMaxScaler.transform(raw_data.reshape(feature_len, 1))
    raw_data = raw_data.reshape(1, 1, feature_len)
    # los_nlos = classify_model.predict(raw_data)
    # los_nlos = los_nlos.reshape(len(los_nlos), 1)
    # los_nlos = MinMaxScaler.inverse_transform(los_nlos)
    los_nlos = 1  # 调试用
    if los_nlos == 1:
        dst_predict = los_pre_model.predict(raw_data)
        dst_predict = dst_predict.reshape(len(dst_predict), 1)
        dst_predict = MinMaxScaler.inverse_transform(dst_predict)
    else:
        dst_predict = nlos_pre_model.predict(raw_data)
        dst_predict = dst_predict.reshape(len(dst_predict), 1)
        dst_predict = MinMaxScaler.inverse_transform(dst_predict)

    # dst_predict = dst_predict.reshape(len(dst_predict), 1)
    print(dst_predict[0][0])
    return dst_predict[0][0], los_nlos


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


def info_collector():
    ser = serial.Serial('COM12', 115200)
    # 设置串口
    distance = 300
    # 设置实际距离，单位mm
    isLoS = 1
    # 设置是否是LoS环境，1为是0为否，当然这里是收集数据用的
    with open('FTM_Predict.csv', 'w', newline='') as file:
        # writer = csv.writer(file)
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
                    # print(data)
                    params = data.split(',')
                    # print(params)
                    # params.append(isLoS)
                    params.append(distance)
                    # writer.writerow(params)
                    apid = params[0]
                    csi = params[5:-1]
                    # print(csi)

                    pred_dist, los_nlos = exec_predict(params[3], params[4], csi)

                    current_ap = APInformation(apid, params[1], params[2], pred_dist, 1, los_nlos)
                    current_ap_ids.append(apid)

                    print("current ap: " + apid)
                    if len(ap_info) == 0:
                        ap_info.append(current_ap)
                    else:
                        exist = 0
                        for i in range(len(ap_info)):
                            if ap_info[i].apid == apid:
                                # ap_info.append(current_ap)
                                ap_info[i].is_valid = 1
                                exist = 1
                                break
                        if exist == 0:
                            ap_info.append(current_ap)

                    if int(apid) >= len(anchor_positions):
                        anchor_positions.append(vec3d(float(params[1]), float(params[2]), 0))
                        dist.append(pred_dist)

                    else:
                        # 一次循环结束

                        for i in range(len(ap_info)):
                            if not current_ap_ids.__contains__(ap_info[i].apid):
                                ap_info[i].is_valid = 0
                        current_ap_ids.clear()
                        for info in ap_info:
                            print("ap: " + str(info.apid) + "isValid: " + str(info.is_valid))

                        distances = np.array(dist)
                        estimated_position = trilateration(anchor_positions, distances)
                        estimated_position = np.round(estimated_position, 3)
                        print("Estimated Position:", estimated_position)
                        anchor_positions.clear()
                        dist.clear()


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>hello world</p>"


@app.route("/ap")
def get_ap_info():
    json_data = json.dumps(ap_info, cls=MyEncoder)
    return json_data


if __name__ == '__main__':
    ap_info = []
    current_ap_ids = []
    print("loading model...")
    los_pre_model = load_model("mlp_model.h5")
    nlos_pre_model = load_model("mlp_model.h5")
    classify_model = load_model("mlp_model.h5")
    MinMaxScaler = joblib.load('mmScaler')
    info_collector_thread = threading.Thread(target=info_collector)
    info_collector_thread.start()
    app.run()
