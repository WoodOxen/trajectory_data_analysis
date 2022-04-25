import sys
sys.path.append("../")
import os
import json
import math
import argparse
import time

import pyproj
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# python3 track_preprocess.py /data/HighD/highD/data highD
# python3 track_preprocess.py /data/HighD/inD/data inD
# python3 track_preprocess.py /data/HighD/rounD/data rounD
# python3 track_preprocess.py /data/INTERACTION/recorded_trackfiles INTERACTION

column_order = ["id", "frame", "type", "length", "width", "x", "y", "v", "angle", "static"]


def interpolate(x, dx, dt):
    return x+0.5*dx*dt*abs(dt)


def inter_speed(df_sub, idx):
    line1 = df_sub.iloc[idx-1]
    line2 = df_sub.iloc[idx]
    vx = (interpolate(line1["xVelocity"], line1["xAcceleration"], 0.02) + interpolate(line2["xVelocity"], line2["xAcceleration"], -0.02)) / 2
    vy = (interpolate(line1["yVelocity"], line1["yAcceleration"], 0.02) + interpolate(line2["yVelocity"], line2["yAcceleration"], -0.02)) / 2
    return vx, vy


def inter_loc(df_sub, idx):
    line1 = df_sub.iloc[idx-1]
    line2 = df_sub.iloc[idx]
    x = (interpolate(line1["x"], line1["xVelocity"], 0.02) + interpolate(line2["x"], line2["xVelocity"], -0.02)) / 2
    y = (interpolate(line1["y"], line1["yVelocity"], 0.02) + interpolate(line2["y"], line2["yVelocity"], -0.02)) / 2
    return x,y


def calibrate_xy(df, df_record_meta, map_config):
    upper_lane_marks = df_record_meta.iloc[0]["upperLaneMarkings"].split(";")
    lower_lane_marks = df_record_meta.iloc[0]["lowerLaneMarkings"].split(";")
    upper_lane_marks = [-float(x) for x in upper_lane_marks]
    lower_lane_marks = [-float(x) for x in lower_lane_marks]
    
    y = np.array(df["y"])
    projector = pyproj.Proj(**map_config["project_rule"])
    utm_origin = projector(map_config["gps_origin"][0],map_config["gps_origin"][1] )
    map_root = ET.parse("../data/maps/%s.osm" % map_config["map_name"])
    pt_y = []
    for node in map_root.findall("node"):
        coordinate = projector(node.get("lon"), node.get("lat"))
        pt_y.append(round(coordinate[1]-utm_origin[1],2))
    pt_y = np.sort(np.unique(pt_y))[::-1]
    
    k = (max(pt_y)-min(pt_y)) / (max(upper_lane_marks)-min(lower_lane_marks))
    b = max(pt_y) - max(upper_lane_marks)*k
    y = np.round(k*y + b,2)
    df["y"] = y
    return df


def change_fps_highD(df_origin):
    agent_ids = df_origin["id"].unique()
    new_id = []
    new_frame = []
    new_x = []
    new_y = []
    width = []
    length = []
    new_vx = []
    new_vy = []
    
    for agent_id in agent_ids:
        df_sub = df_origin[df_origin["id"] == agent_id].reset_index()
        for idx, line in df_sub.iterrows():
            if  (line["frame"] - 1) % 5 == 0:
                new_id.append(int(line["id"]))
                new_frame.append(int((line["frame"])/2.5+1))
                new_x.append(round(line["x"] + line["width"]/2, 2))
                new_y.append(round(-line["y"] - line["height"]/2, 2))
                length.append(line["width"])
                width.append(line["height"])
                new_vx.append(line["xVelocity"])
                new_vy.append(-line["yVelocity"])
            elif (idx > 0) and ((line["frame"]-4) % 5 == 0):
                new_id.append(int(line["id"]))
                new_frame.append(int((line["frame"]+1) / 2.5))
                length.append(line["width"])
                width.append(line["height"])
                vx, vy = inter_speed(df_sub, idx)
                new_vx.append(round(vx, 2))
                new_vy.append(round(-vy, 2))
                x, y = inter_loc(df_sub, idx)
                new_x.append(round(x+line["width"]/2, 2))
                new_y.append(round(-y-line["height"]/2, 2))
    
    df = pd.DataFrame({
        "id": new_id,
        "frame": new_frame,
        "length": length,
        "width":  width,
        "x": new_x,
        "y": new_y,
    })
                
    return df, new_vx, new_vy


def inter_angle(df, idx):
    return (df.iloc[idx]["heading"] + df.iloc[idx-1]["heading"]) / 2


def inter_loc_inD(df_sub, idx):
    line1 = df_sub.iloc[idx-1]
    line2 = df_sub.iloc[idx]
    x = (interpolate(line1["xCenter"], line1["xVelocity"], 0.02) + interpolate(line2["xCenter"], line2["xVelocity"], -0.02)) / 2
    y = (interpolate(line1["yCenter"], line1["yVelocity"], 0.02) + interpolate(line2["yCenter"], line2["yVelocity"], -0.02)) / 2
    return x,y


def change_fps(df_origin, df_meta):
    df_meta = df_meta.sort_values(by="initialFrame").reset_index()
    agent_ids = np.array(df_meta["trackId"])
    
    new_id = []
    new_frame = []
    new_x = []
    new_y = []
    width = []
    length = []
    new_vx = []
    new_vy = []
    new_angle = []
    
    for agent_id in agent_ids:
        df_sub = df_origin[df_origin["trackId"] == agent_id].reset_index()
        for idx, line in df_sub.iterrows():
            if  line["frame"] % 5 == 0:
                new_id.append(int(line["trackId"]))
                new_frame.append(int((line["frame"])/2.5))
                new_x.append(round(line["xCenter"], 2))
                new_y.append(round(line["yCenter"], 2))
                width.append(round(line["width"], 2))
                length.append(round(line["length"], 2))
                new_vx.append(round(line["xVelocity"], 2))
                new_vy.append(round(line["yVelocity"], 2))
                new_angle.append(line["heading"])
            elif (idx > 0) and ((line["frame"]-3) % 5 == 0):
                new_id.append(int(line["trackId"]))
                new_frame.append(int((line["frame"]+1) / 2.5))
                width.append(round(line["width"], 2))
                length.append(round(line["length"], 2))
                vx, vy = inter_speed(df_sub, idx)
                new_vx.append(round(vx, 2))
                new_vy.append(round(vy, 2))
                x, y = inter_loc_inD(df_sub, idx)
                new_x.append(round(x, 2))
                new_y.append(round(y, 2))
                new_angle.append(inter_angle(df_sub, idx))
                
    speed = np.round(np.sqrt(np.square(new_vx) + np.square(new_vy)), 2)
    new_angle = np.round(np.array(new_angle)/360*2*math.pi, 3)
    
    df_processed = pd.DataFrame({
        "id": new_id,
        "frame": new_frame,
        "length": length,
        "width": width,
        "x": new_x,
        "y": new_y,
        "v": speed,
        "angle": new_angle
    })
    return df_processed


def get_type(df, df_meta, idx_name):
    type = []
    for _, line in df.iterrows():
        type.append(df_meta[df_meta[idx_name]==line["id"]].iloc[0]["class"])
    return type


def get_velocity(vx, vy):
    """Get the velocity vector and return them in the form of speed and angle
    """
    speed = []
    angle = []
    for v in zip(vx, vy):
        speed.append(round(np.sqrt(v[0]**2+v[1]**2), 3))
        angle.append(round(math.atan2(v[1], v[0]), 3))
    return speed, angle


def get_displacement_upper_bound(df):
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    displacement = np.sqrt((x_min - x_max)**2 + (y_min- y_max)**2)
    return displacement


def get_static_label(df, threshold=2):
    agent_ids = df["id"].unique()
    is_static = []
    
    for agent_id in agent_ids:
        df_sub = df[df["id"] == agent_id]
        disp_max = get_displacement_upper_bound(df_sub)
        is_static += [1 if disp_max<threshold else 0] * len(df_sub)
    return is_static


def process_highD(args):
    dir_config = "../data/configs/%s.config" % args.dataset
    if args.config is not None:
        dir_config = args.config
    config = json.load(open(dir_config, "r"))
    
    for map_id, map_config in config.items():
        dir_out = "%s/%s" % (args.out, map_config["map_name"])
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        for track_id in map_config["track_files"]:
            t1 = time.time()
            file_out = "%s/%03d_tracks.csv" % (dir_out, track_id)
            if os.path.exists(file_out) and (not args.overwrite):
                continue
            
            df = pd.read_csv(os.path.join(args.dir, "%02d_tracks.csv" % track_id))
            df_meta = pd.read_csv(os.path.join(args.dir, "%02d_tracksMeta.csv" % track_id))
            df_record_meta = pd.read_csv(os.path.join(args.dir, "%02d_recordingMeta.csv" % track_id))
            
            df_processed, new_vx, new_vy = change_fps_highD(df)
            df_processed = calibrate_xy(df_processed, df_record_meta, map_config)
            df_processed["type"] = get_type(df_processed, df_meta, "id")
            df_processed["v"], df_processed["angle"] = get_velocity(new_vx, new_vy)
            df_processed["static"] = get_static_label(df_processed)
            df_processed = df_processed[column_order]
            
            df_processed.to_csv(file_out, index=False)
            t2 = time.time()
            print("Finish track file %03d in %s, using %f s." % (track_id, args.dataset, t2-t1))


def process_inD_rounD(args):
    dir_config = "../data/configs/%s.config" % args.dataset
    if args.config is not None:
        dir_config = args.config
    config = json.load(open(dir_config, "r"))
    
    for map_id, map_config in config.items():
        dir_out = "%s/%s" % (args.out, map_config["map_name"])
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        for track_id in map_config["track_files"]:
            t1 = time.time()
            file_out = "%s/%03d_tracks.csv" % (dir_out, track_id)
            if os.path.exists(file_out) and (not args.overwrite):
                continue
            
            df = pd.read_csv(os.path.join(args.dir, "%02d_tracks.csv" % track_id))
            df_meta = pd.read_csv(os.path.join(args.dir, "%02d_tracksMeta.csv" % track_id))
            
            df_processed = change_fps(df, df_meta)
            df_processed["type"] = get_type(df_processed, df_meta, "trackId")
            df_processed["static"] = get_static_label(df_processed)
            df_processed = df_processed[column_order]
            df_processed.to_csv(file_out, index=False)
            t2 = time.time()
            print("Finish track file %03d in %s, using %f s." % (track_id, args.dataset, t2-t1))
    

def ped_type(df):
    ped_ids = np.unique(df["track_id"])
    type = []
    for ped_id in ped_ids:
        df_sub = df[df["track_id"]==ped_id]
        ped_v = np.round(np.sqrt(np.square(df_sub["vx"])+np.square(df_sub["vy"])), 2)
        if (np.mean(ped_v) < 2) and (np.max(ped_v) < 2.5):
            type += ["pedestrian"] * len(df_sub)
        else:
            type += ["bicycle"] * len(df_sub)
    return type


def process_INTERACTION(args):
    dir_config = "../data/configs/%s.config" % args.dataset
    if args.config is not None:
        dir_config = args.config
    config = json.load(open(dir_config, "r"))

    for _, map_config in config.items():
        dir_out = "%s/%s" % (args.out, map_config["map_name"])
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        track_file_list = os.listdir("%s/%s/" % (args.dir, map_config["original_map_name"]))
        
        id_list = []
        for track_file_name in track_file_list:
            id_list.append(int(track_file_name[-7:-4]))
        id_list = np.unique(id_list)
        
        for track_id in id_list:
            t1 = time.time()
            file_out = "%s/%03d_tracks.csv" % (dir_out, track_id)
            if os.path.exists(file_out) and (not args.overwrite):
                continue
            df = pd.read_csv("%s/%s/vehicle_tracks_%03d.csv" % (args.dir, map_config["original_map_name"], track_id))
            df_processed = pd.DataFrame({
                "id": df["track_id"],
                "frame": df["frame_id"],
                "type": df["agent_type"],
                "length": df["length"],
                "width": df["width"],
                "x": np.round(df["x"], 2),
                "y": np.round(df["y"], 2),
                "v": np.round(np.sqrt(np.square(df["vx"])+np.square(df["vy"])), 2),
                "angle": np.round(df["psi_rad"], 3)
            })
                
            if os.path.exists("%s/%s/pedestrian_tracks_%03d.csv" % (args.dir, map_config["original_map_name"], track_id)):
                df_pedestrian = pd.read_csv("%s/%s/pedestrian_tracks_%03d.csv" % (args.dir, map_config["original_map_name"], track_id))
                n = len(df_pedestrian)
                df_ped_processed = pd.DataFrame({
                    "id": df_pedestrian["track_id"],
                    "frame": df_pedestrian["frame_id"],
                    "type": ped_type(df_pedestrian),
                    "length": [-1] * n,
                    "width": [-1] * n,
                    "x": np.round(df_pedestrian["x"], 2),
                    "y": np.round(df_pedestrian["y"], 2),
                    "v": np.round(np.sqrt(np.square(df_pedestrian["vx"])+np.square(df_pedestrian["vy"])), 2),
                    "angle": np.round(np.arctan2(df_pedestrian["vy"], df_pedestrian["vx"]),3)
                })
                df_processed = pd.concat([df_processed, df_ped_processed])

            df_processed["static"] = get_static_label(df_processed)
            df_processed = df_processed[column_order]
            df_processed.to_csv(file_out, index=False)
            t2 = time.time()
            print("Finish track file %03d in %s, using %f s." % (track_id, args.dataset, t2-t1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=""
    )
    parser.add_argument(
        "dir",
        help="path to the dataset"
    )
    parser.add_argument(
        "dataset", choices=["highD", "inD", "rounD", "INTERACTION"],
        help="name of the dataset"
    )
    parser.add_argument(
        "-out", default="../data/tracks", required=False,
        help="path to processed track files"
    )
    parser.add_argument(
        "-config", required=False,
        help="path to the dataset's configuration file"
    )
    parser.add_argument(
        "-overwrite", action="store_true", default=False, required=False,
        help="overwriting those existing processed track files"
    )
    
    args = parser.parse_args()

    
    if args.dataset == "highD":
        process_highD(args)
    elif args.dataset in ["inD", "rounD"]:
        process_inD_rounD(args)
    elif args.dataset == "INTERACTION":
        process_INTERACTION(args)