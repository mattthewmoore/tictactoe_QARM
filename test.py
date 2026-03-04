def get_stable_dept_data(self,centroid_arr, depth_frame):
    valid_readings = []
    depth = 0.0
    range = 5
    cx = centroid_arr(0)
    cy = centroid_arr(1)

    for i in range (-range,range+1):
        for j in range(-range, range+1):
            x = cx + i
            y = cy + j
            
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                d = depth_frame[x,y]

                if d > 0:
                    valid_readings.append(d)
    average = float(np.mean(valid_readings))
    if valid_readings:
        depth = average
    return depth