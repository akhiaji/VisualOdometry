from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
sift = cv.xfeatures2d.SIFT_create()
from pyquaternion import Quaternion
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import argparse




def calcTransform(img2, img1, h2, h1, fx, fy, cx, cy):
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    img1_idx = []
    img2_idx = []
    img1_points = []
    img2_points = []
    X_1 = np.array([0.0, 0.0, 0.0])
    X_2 = np.array([0.0, 0.0, 0.0])
    samples = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            x,y = kp1[m.queryIdx].pt
            r1, c1 = int(y), int(x)
            x,y = kp2[m.trainIdx].pt
            r2, c2 = int(y), int(x)
            if h1[r1,c1] != 0 and h2[r2,c2] != 0:
                img1_idx.append(m.queryIdx)
                x = kp1[m.queryIdx].pt[0]
                y = kp1[m.queryIdx].pt[1]
                img1_points.append([(x - cx) * h1[r1,c1]/ fx, (y - cy) * h1[r1,c1]/ fy, h1[r1,c1]])
                img2_idx.append(m.trainIdx)
                x = kp2[m.trainIdx].pt[0]
                y = kp2[m.trainIdx].pt[1]
                img2_points.append([(x - cx) * h2[r2,c2]/ fx, (y - cy) * h2[r2,c2]/ fy, h2[r2,c2]])
                samples += 1
    X1_points = np.matrix(img1_points).T
    X2_points = np.matrix(img2_points).T
    X_1 = np.average(X1_points, axis=0)
    X_2 = np.average(X2_points, axis=0)
    R = np.eye(3)
    for i in range(1):
        X1_points = R@X1_points
        X2_points = X2_points
        X_1 = np.average(X1_points, axis=1).reshape(-1, 1)
        X_2 = np.average(X2_points, axis=1).reshape(-1, 1)
        SVD_input = (X1_points-X_1)[0:3] @ (X2_points-X_2)[0:3].T
        U,S,V_t = np.linalg.svd(SVD_input)
        R = V_t.T @ U.T @ R
        if np.linalg.det(R) < 0:
            V = V_t.T
            V[:,2] *= -1
            R = V @ U.T @ R
        t = X_2[0:3] - R @ X_1[0:3]
        t2 = np.zeros((4,4))
        t2[0:3,0:3] = R
        t2[0:3,3] = t.reshape(-1)
        t2[3,3] = 1
    return t2

def ICP(path, fx, fy, cx, cy, x, y, z, q1, q2, q3, q4):
    rgb_path = path + '/rgb/'
    rgb_files = sorted([f for f in listdir(rgb_path) if isfile(join(rgb_path, f))])
    depth_path = path + '/depth/'
    depth_files = sorted([f for f in listdir(depth_path) if isfile(join(depth_path, f))])
    last_rgb = None
    last_h = None
    translation = np.array([x, y, z, 1])
    rotation = Quaternion(q1, q2, q3, q4)
    position = np.zeros((4,4))
    position[0:3,0:3] = rotation.rotation_matrix
    position[:, 3] = translation
    positions = []
    output_data = []
    i = 0

    f, ax = plt.subplots(2)

    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    for rgb_name, h_name in zip(rgb_files, depth_files):
        i += 1
        if last_rgb is None and last_h is None:
            last_rgb = cv.imread(rgb_path+rgb_name,0)
            last_h = (cv.imread(depth_path+h_name,-cv.IMREAD_ANYDEPTH)/5000)*1.035
        else:
            rgb = cv.imread(rgb_path+rgb_name,0)
            h = (cv.imread(depth_path+h_name,-cv.IMREAD_ANYDEPTH)/5000)*1.035
            transformation = calcTransform(last_rgb, rgb, last_h, h, fx, fy, cx, cy)
            if np.linalg.det(transformation[0:3, 0:3]) < 0:
                print(i)
            position = position @ transformation

            # real time plotting
            draw_x, draw_y = position[0:2, 3]
            ax[0].scatter(draw_x, draw_y, c = 'blue')
            ax[1].imshow(rgb)
            plt.pause(0.03)

            try:
                r = Quaternion(matrix=position[0:3, 0:3])
                rotation = r
            except ValueError:
                print("Rotation Error")
                position[0:3,0:3] = rotation.rotation_matrix

            positions.append((position[0:3, 3], rotation.elements))

            output = np.zeros((8))
            output[0] = float(rgb_name[:-4])
            output[1:4] = position[0:3, 3]
            output[4:8] = rotation.elements[:]
            output_data.append(output)
            last_rgb = rgb
            last_h = h
        if i % 50 == 0:
            print(i)
    
    #plt.show()
    return positions, output_data

def errorPerStep(outputData, groundtruth, aligned=False):
    df = pd.DataFrame(outputData)
    df = df.round(6)
    # df.to_csv("results.txt", header=False, index=False, sep=' ')
    gt = pd.read_csv(groundtruth, header=None, delim_whitespace=True, comment='#')



    potential_matches = [(abs(a - (b)), a, b)
                             for a in df[0]
                             for b in gt[0]
                             if abs(a - (b)) < .02]
    first_keys = list(df[0])
    second_keys = list(gt[0])
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()

    first_list = df.set_index(0)
    second_list = gt.set_index(0)

    first_xyz = np.matrix([[float(value) for value in list(first_list.loc[a])[0:3]] for a,b in matches]).transpose()
    second_xyz = np.matrix([[float(value) for value in list(second_list.loc[b])[0:3]] for a,b in matches]).transpose()
    rot,trans,trans_error = align(second_xyz,first_xyz)
    if aligned:
        rot,trans,trans_error = align(second_xyz,first_xyz)
        second_xyz = rot * second_xyz + trans

    first_stamps = np.array(first_list.index)
    first_stamps.sort()
    first_xyz_full = np.matrix([[float(value) for value in first_list.loc[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = np.array(second_list.index)
    second_stamps.sort()
    second_xyz_full = np.matrix([[float(value) for value in second_list.loc[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans
    if aligned:
        rot,trans,trans_error = align(second_xyz,first_xyz)
        second_xyz_full = rot * second_xyz_full + trans
    error = []
    total_error = [0]

    for i in range(second_xyz.shape[1]-1):
#         print("recovered delta", second_xyz[:,i+1]-second_xyz[:,i])
#         print("truth delta", first_xyz[:,i+1]-first_xyz[:,i])
        error.append(np.sum(np.abs((second_xyz[:,i+1]-second_xyz[:,i]) - (first_xyz[:,i+1]-first_xyz[:,i]) )))
        total_error.append(total_error[-1]+error[i])
    print("compared_pose_pairs %d pairs"%(len(trans_error)))

    print("absolute_translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
    print("absolute_translational_error.mean %f m"%np.mean(trans_error))
    print("absolute_translational_error.median %f m"%np.median(trans_error))
    print("absolute_translational_error.std %f m"%np.std(trans_error))
    print("absolute_translational_error.min %f m"%np.min(trans_error))
    print("absolute_translational_error.max %f m"%np.max(trans_error))
    return error, total_error

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
    return rot,trans,trans_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Script runs Visual Odometry with the appropriate settings
    ''')
    parser.add_argument('--path', help='directory with the data', default='rgbd_dataset_freiburg1_floor', type=str)
    parser.add_argument('--fx', help='focal length fx', default=517.3, type=float)
    parser.add_argument('--fy', help='focal length fy', default=516.5, type=float)
    parser.add_argument('--cx', help='optical center cx', default=318.6, type=float)
    parser.add_argument('--cy', help='optical center cy', default=255.3, type=float)
    parser.add_argument('--x', help='initial x position', default= 1.2764, type=float)
    parser.add_argument('--y', help='initial y position', default= -0.9763, type=float)
    parser.add_argument('--z', help='initial z position', default= 0.6837, type=float)
    parser.add_argument('--q1', help='first initial quaternion component', default= 0.8187, type=float)
    parser.add_argument('--q2', help='seond initial quaternion component', default= 0.3639, type=float)
    parser.add_argument('--q3', help='third initial quaternion component', default= -0.1804, type=float)
    parser.add_argument('--q4', help='fourth initial quaternion component', default= -0.4060, type=float)
    args = parser.parse_args()

    positions, output_data = ICP(args.path, args.fx, args.fy, args.cx, args.cy, args.x, args.y, args.z, args.q1, args.q2, args.q3, args.q4)
    df = pd.DataFrame(output_data)
    df = df.round(6)
    df.to_csv(args.path.split('/')[-1] + "_results.txt", header=False, index=False, sep=' ')
    error, error_summed = errorPerStep(output_data, args.path + "/groundtruth.txt", True)
    f, ax = plt.subplots(2,2)
    df = pd.read_csv(args.path + "/groundtruth.txt", header=None, delim_whitespace=True, comment='#')
    df = df[[1,2,3]]
    p = np.array(df)
    x = p.T[0][1::4]
    y = p.T[1][1::4]
    c = range(len(x))
    ax[0][0].scatter(x,y,c=c, cmap='viridis')
    ax[0][0].set_title("Ground Truth")
    pos_points = np.array([i[0] for i in positions])
    x = np.array(pos_points).T[0]
    y = np.array(pos_points).T[1]
    z = np.array(pos_points).T[2]
    c = range(len(x))
    ax[1][0].set_title("Visual Odom")
    ax[1][0].scatter(x, y,c=c,cmap='viridis')
    ax[0][1].plot(range(len(error_summed)), error_summed)
    ax[0][1].set_title("Accumulated Error")
    ax[1][1].plot(range(len(error)), error)
    ax[1][1].set_title("Error Per Step")
    f.savefig(args.path.split('/')[-1] + "_charts")
    f.show()
