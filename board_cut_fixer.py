import math
import cv2
import identify_board
import copy
import numpy as np
import bisect
import board_cut_checker

# import gui_img_manager

### DEBUG FLAG ###
DEBUG = False

##### Gaussian Threshold parameters #####
GAUSS_MAX_VALUE = 255
GAUSS_BLOCK_SIZE = 115
GAUSS_C = 13


##### Edge detection parameters #####
EDGE_DST = 10
EDGE_KSIZE = 3
EDGE_SCALE = 10

MIN_GRID_SIZE = 1.0 / 20
MAX_GRID_SIZE = 1.0 / 8
MAX_LINE_DIST_RATIO = 1.0 / 15
RESIZE_HEIGHT = identify_board.RESIZE_HEIGHT
RESIZE_WIDTH = identify_board.RESIZE_WIDTH
MAX_NUM_LINES = 11
MAX_LINES_IN_GRID = 9
LINE_SERIES_COEFF = 5
LINE_SERIES_POWER = 1
ANGLE_STD_COEFF = 0.8
MIN_NUM_ANGLES_AVG = 3

### Filter lines by surrounding color parameters ###
MIN_MIXED_AREA_VAR_LIGHTNESS = 15
MIXED_AREA = 0
UNIFORM_AREA = 1
INSIDE_LINE = 0
OUTSIDE_LINE = 1
BAD_LINE = 2

LINE_METRIC_ANGLE_ERROR_COEFF = 150
LINE_METRIC_DIST_ERROR_COEFF = 6

LINE_UNIFORM_SCANNER_PIXEL_JUMP = 15

#####sizes of the window in the convolution
PartOfWindowHeight = 16
PartOfWindowLength = 60
ConvSkipX = 4
ConvSkipY = 4

#####number of changes to be an legal image
CHANGE_COLOR_SAF = 6
CHANGE_DIFF = 6
BOTTOM_MAX_CHANGE_BEL = 3
TOP_MAX_CHANGE_ABV = 3

##### Gaussian Threshold parameters #####
GAUSS_MAX_VALUE = 255
GAUSS_BLOCK_SIZE = 109
GAUSS_C = 13

##### nuber of iterations for the board cut fixer#####
NUM_ITERATIONS = 2
NUM_ANGLE_ITERATIONS = 2
ANGLE_FIX = 1
MIN_SERIES_SCORE = 6

# Line-finding parameters
VER_ANGLE_RANGE = 0.1
HOR_ANGLE_RANGE = 0.18
ANGLE_CONSTRAINT_DECAY_FACTOR = 0.3

PROJECTION_SPARE_GRID_SIZE = 14
PROJECTION_SPARE_DIFF = (PROJECTION_SPARE_GRID_SIZE - 8) / 2
PROJECTION_SPARE_GRID_SIZE_BIG = 16
PROJECTION_SPARE_DIFF_BIG = (PROJECTION_SPARE_GRID_SIZE_BIG-8)/2

## for getting best bottom line - number of options to test
BOTTOM_LINE_INDEX_VARIATION = 3

## for image diff area
IM_DIFF_AREA_SKIP = 7

FINAL_IMG_WIDTH = 400
FINAL_IMG_HEIGHT = 450

class board_cut_fixer:
    def __init__(self):
        self.last_image_bw = cv2.imread("utils\\img_start_bw.jpg",
                                        cv2.IMREAD_GRAYSCALE)
        self.board_id = identify_board.identify_board()

    def set_prev_im(self,im):
        self.last_image_bw = self.gausThresholdChess(im)
        wid = len(im[0])
        hi = len(im)
        self.last_image_bw = self.last_image_bw[hi//9:hi,0:wid]
        self.last_image_bw = cv2.resize(self.last_image_bw, (RESIZE_WIDTH,RESIZE_HEIGHT))


    def get_theta(self, line):
        try:
            if line[2] == line[0]:
                theta = math.pi / 2
            else:
                theta = (float)(math.atan2(float(line[3] - line[1]), float(line[2] - line[0])))
        except:
            print('1'
                  '')
        return (theta % math.pi);

        """"""

    def draw_points(self, img, points):
        if (DEBUG):
            img = copy.deepcopy(img)
            for point in points:
                fix_point = (int(point[0]), int(point[1]))
                cv2.circle(img, fix_point, 10, [0, 255, 255])
            cv2.imshow('image', img)
            cv2.waitKey(0)

    """
    rounded modulo
    """

    def modulo(self, x, y):

        mod = x % y
        if mod > y / 2:
            mod = y - mod
        return mod

    def get_board_limits(self, im, points):
        board_size = len(im)
        best_pair = self.get_grid_origin_pair(points, board_size)
        p1 = best_pair[0]
        p2 = best_pair[1]
        self.draw_points(im, best_pair)
        axes = self.get_axes_from_pair(best_pair)
        x_axis = axes[0]
        y_axis = axes[1]
        row_scores = {}
        col_scores = {}
        for i in range(-MAX_NUM_LINES, MAX_NUM_LINES + 1):
            row_scores[i] = 0
            col_scores[i] = 0
        for p in points:
            grid_loc = self.get_point_grid_location(p1, x_axis, y_axis, p)
            score = self.get_point_grid_metric(grid_loc)
            row = round(grid_loc[0])
            col = round(grid_loc[1])
            row_scores[row] = row_scores[row] + score
            col_scores[col] = col_scores[col] + score

        max_row_score = 0
        max_col_score = 0
        left_row_idx = -12
        top_row_idx = -12
        for i in range(-MAX_NUM_LINES, 1):
            row_score = (row_scores[i]) ** 0.5 + (row_scores[i + 8]) ** 0.5
            col_score = (col_scores[i]) ** 0.5 + (col_scores[i + 8]) ** 0.5
            if (row_score > max_row_score):
                max_row_score = row_score
                left_row_idx = i
            if (col_score > max_col_score):
                max_col_score = col_score
                top_col_idx = i

        ul_pt = (p1[0] + left_row_idx * x_axis[0] + top_col_idx * y_axis[0],
                 p1[1] + left_row_idx * x_axis[1] + top_col_idx * y_axis[1])

        br_pt = (p1[0] + (left_row_idx + 8) * x_axis[0] + (top_col_idx + 8) * y_axis[0],
                 p1[1] + (left_row_idx + 8) * x_axis[1] + (top_col_idx + 8) * y_axis[1])

        ur_pt = (p1[0] + (left_row_idx + 8) * x_axis[0] + top_col_idx * y_axis[0],
                 p1[1] + (left_row_idx + 8) * x_axis[1] + top_col_idx * y_axis[1])

        bl_pt = (p1[0] + left_row_idx * x_axis[0] + (top_col_idx + 8) * y_axis[0],
                 p1[1] + left_row_idx * x_axis[1] + (top_col_idx + 8) * y_axis[1])
        return (ul_pt, ur_pt, br_pt, bl_pt)

    def Make_3d_List_2_2d_List(self, list):
        list2d = []
        for li in list:
            for l in li:
                list2d.append(l)
        return list2d

    def connectLines(self, lines, board_size):
        max_dis = board_size * MAX_LINE_DIST_RATIO
        theta = self.get_theta(lines[0])
        new_lines = []
        values = []
        new_lines.append(lines[0])
        if theta < 0.03 or theta > 3.1:
            values.append(lines[0][1])
            for i in range(1, len(lines)):
                line = lines[i]
                added = True
                for j in range(len(values)):
                    if abs(line[1] - values[j]) < 20:
                        added = False
                if added:
                    new_lines.append(line)
                    values.append(line[1])
        if theta < (math.pi / 2) + 0.03 or theta > (math.pi / 2) - 0.03:
            values.append(lines[0][0])
            for i in range(1, len(lines)):
                line = lines[i]
                added = True
                for j in range(len(values)):
                    if abs(line[0] - values[j]) < max_dis:
                        added = False
                if added:
                    new_lines.append(line)
                    values.append(line[0])
        return new_lines

    def get_cutoff_point(self, line1, line2):
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]
        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]
        if x1 == x2:
            m1 = 10000000000
        else:
            m1 = float(float(y2 - y1) / float(x2 - x1))
        if x3 == x4:
            m2 = 10000000000
        else:
            m2 = float(float(y4 - y3) / float(x4 - x3))
        n1 = y1 - m1 * x1
        n2 = y3 - m2 * x3

        x = int((n1 - n2) / (m2 - m1))
        y = int(m1 * x + n1)
        point = []
        point.append(x)
        point.append(y)
        return point


    def rotate_image_fix(self, image):
        rows,cols = image.shape[0:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ANGLE_FIX, 1)
        return cv2.warpAffine(image, M, (cols,rows))

    """
    fixes projected image
    """

    def projection(self, pointslst, img, frame, take_spare, big_picture):

        if (take_spare):
            diff = PROJECTION_SPARE_DIFF
            gridsize = PROJECTION_SPARE_GRID_SIZE
        else:
            diff = 0
            gridsize = 8
        if(big_picture):
            diff = PROJECTION_SPARE_DIFF_BIG
            gridsize = PROJECTION_SPARE_GRID_SIZE_BIG
        pts1 = np.float32(pointslst)
        x_hi = (diff + frame[1]) * RESIZE_WIDTH / gridsize
        x_lo = (diff + frame[3]) * RESIZE_WIDTH / gridsize
        y_hi = (8 + diff - frame[0]) * RESIZE_HEIGHT / gridsize
        y_lo = (8 + diff - frame[2]) * RESIZE_HEIGHT / gridsize
        pts2 = np.float32([[x_lo, y_hi], [x_hi, y_hi],
                           [x_hi, y_lo], [x_lo, y_lo]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # if (DEBUG):
        #    cv2.imshow("image", dst)
        #    k = cv2.waitKey(0)
        return dst

    def draw_lines(self, lineslst, img):
        if (DEBUG):
            new_img = copy.deepcopy(img)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(lineslst)):
                l = lineslst[i]
                cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (255, 255,
                                                               0), 3,
                         cv2.LINE_AA)
            # img = cv2.resize(img, (0, 0), fx=1, fy=1)
            cv2.imshow('image', new_img)
            k = cv2.waitKey(0)
            return

    def get_lines(self, img, constraint_factor):
        linesP = cv2.HoughLinesP(img, 1, math.pi / 180, 100,
                                 minLineLength=100, maxLineGap=50)
        lines = self.Make_3d_List_2_2d_List(linesP)
        hor = []
        ver = []
        ver_range = VER_ANGLE_RANGE * constraint_factor
        hor_range = HOR_ANGLE_RANGE * constraint_factor
        for l in lines:
            theta = self.get_theta(l)
            if theta < (math.pi / 2) + ver_range and \
                            theta > (math.pi / 2) - ver_range:
                ver.append(l)
            if theta < hor_range or theta > math.pi - hor_range:
                hor.append(l)
                #        hor.sort(key=lambda x: x[1], reverse=True)
        # hor = self.connectLines(hor,len(img))
        # ver = self.connectLines(ver,len(img))
        return hor, ver

    """
    given a list of lines, and a function that gives a value for each line,
    find the most accurate item in the series, and the delta value of the
    series.
    :return a list of best fits to the series, SORTED BY FIT!
    """

    def get_im_from_bigim(self, bigim):
       # if(spare):
        x_low = len(bigim[0])*PROJECTION_SPARE_DIFF_BIG/PROJECTION_SPARE_GRID_SIZE_BIG
        x_hi = len(bigim[0])*(1-PROJECTION_SPARE_DIFF_BIG/PROJECTION_SPARE_GRID_SIZE_BIG)

        y_low = len(bigim) * PROJECTION_SPARE_DIFF_BIG / PROJECTION_SPARE_GRID_SIZE_BIG
        y_hi = len(bigim) * (
    1 - PROJECTION_SPARE_DIFF_BIG / PROJECTION_SPARE_GRID_SIZE_BIG)

        pts = [[x_low,y_low],[x_hi,y_low],[x_hi,y_hi], [x_low,y_hi]]
        frame = [8,8,0,0]

        return self.projection(pts,bigim,frame,False,False)

    def get_line_series(self, lines, valfunc, lower_d, upper_d, num_vals,
                        ):
        lines.sort(key=valfunc)
        vals = [valfunc(l) for l in lines]
        best_d = 0
        best_score = 0
        best_lines = []
        best_line_index = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if not i == j:  # delta val cannot be 0.
                    d = abs(vals[j] - vals[i])
                    if (lower_d <= d <= upper_d):
                        start = -(num_vals - 1)
                        end = num_vals - 1
                        scores = []
                        tmp_lines = []
                        for n in range(start, end + 1):

                            val_n = vals[i] + n * d
                            if (val_n < -d or val_n > RESIZE_WIDTH + d):  # exit if
                                # beyond bounds
                                scores.append(0)
                                tmp_lines.append(lines[i])
                            else:
                                # find closest value <= val_n
                                val_closest_idx = bisect.bisect_right(vals,
                                                                      val_n) - 1
                                if (val_closest_idx + 1) < len(vals) and vals[val_closest_idx + 1] - val_n < \
                                                val_n - vals[val_closest_idx]:
                                    val_closest_idx = val_closest_idx + 1
                                val_closest = vals[val_closest_idx]
                                line_closest = lines[val_closest_idx]
                                diff = abs(val_n - val_closest)
                                if diff <= d / 2:
                                    scores.append(1 / (
                                        1 + LINE_SERIES_COEFF * diff * 1.0 / d) ** LINE_SERIES_POWER)
                                    tmp_lines.append(line_closest)
                                else:
                                    scores.append(0)
                                    tmp_lines.append(lines[i])
                                    # score it
                                    # will
                                    #  have :(

                        score = sum(scores[0:num_vals])
                        best_window_score = score
                        offset = 0
                        for k in range(0, len(scores) - num_vals):
                            score = score + scores[num_vals + k] - scores[k]
                            if score > best_window_score:
                                best_window_score = score
                                offset = k + 1

                        if best_window_score > best_score:
                            best_d = d
                            best_score = best_window_score
                            best_lines = tmp_lines[offset:offset + num_vals]
                            best_line_index = num_vals - offset - 1
        if (DEBUG):
            print("best linear series:")
            print("d/boardsize = " + str(best_d / RESIZE_WIDTH))
            print("score: " + str(best_score))

        if best_score<MIN_SERIES_SCORE:
            print("Error: Best series too bad")
            raise Exception()

        return best_lines, best_line_index, best_d

    def line_eq(self, l1, l2):
        return l1[0] == l2[0] and l1[1] == l2[1] and l1[2] == l2[2] and l1[3] == l2[3]

    def make_hor_line(self, point, angle):
        x1 = point[0]
        y1 = point[1]
        x2 = RESIZE_WIDTH - 1
        y2 = math.atan(angle) * (x2 - x1) + y1
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return [x1, y1, x2, y2]

    def make_ver_line(self, point, angle):
        x1 = point[0]
        y1 = point[1]
        y2 = RESIZE_HEIGHT - 1
        x2 = math.atan(-angle + math.pi / 2) * (y2 - y1) + x1
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return [x1, y1, x2, y2]

    def get_board_limits(self, up_line, right_line, bot_line, left_line):
        points = []
        points.append(self.get_cutoff_point(up_line, left_line))
        points.append(self.get_cutoff_point(up_line, right_line))
        points.append(self.get_cutoff_point(bot_line, right_line))
        points.append(self.get_cutoff_point(bot_line, left_line))
        return points

    """
    check if line's area contains black&white.
    """

    def get_area_type(self, line, img, above):
        dx = 5
        dy = 5
        if (above):
            x_min = max(min(line[0], line[2]) - dx, 0)
            x_max = min(max(line[0], line[2]) + dx, len(img[0]))
            y_min = max(min(line[1], line[3]), 0)
            y_max = min(max(line[1], line[3]) + dy, len(img))
        else:
            x_min = max(min(line[0], line[2]) - dx, 0)
            x_max = min(max(line[0], line[2]) + dx, len(img[0]))
            y_min = max(min(line[1], line[3]) - dy, 0)
            y_max = min(max(line[1], line[3]), len(img))
        avg_lightness = 0
        for i in range(x_min, x_max, LINE_UNIFORM_SCANNER_PIXEL_JUMP):
            for j in range(y_min, y_max):
                avg_lightness += max(img[j][i]) / 2 + min(img[j][i]) / 2

        avg_lightness = avg_lightness * 1.0 / (len(range(x_min, x_max,
                                                         LINE_UNIFORM_SCANNER_PIXEL_JUMP)) * (y_max - y_min))
        var_lightness = 0
        for i in range(x_min, x_max, LINE_UNIFORM_SCANNER_PIXEL_JUMP):
            for j in range(y_min, y_max):
                var_lightness += abs(max(img[j][i]) / 2 + min(img[j][i]) / 2 - avg_lightness)
        var_lightness = var_lightness * 1.0 / (len(range(x_min, x_max, LINE_UNIFORM_SCANNER_PIXEL_JUMP)) * (
            y_max - y_min))
        if var_lightness > MIN_MIXED_AREA_VAR_LIGHTNESS:
            return MIXED_AREA
        return UNIFORM_AREA

    """
    color img plz
    """

    def get_lines_types(self, lines, img):
        types = [(self.get_area_type(line, img, True), self.get_area_type(line, img, False)) for line in lines]
        bad = []
        out = []
        ins = []
        for i in range(len(types)):
            if types[i][0] == MIXED_AREA and types[i][1] == MIXED_AREA:
                ins.append(lines[i])
            elif types[i][0] == MIXED_AREA and types[i][1] == UNIFORM_AREA:
                out.append(lines[i])
            elif types[i][0] == UNIFORM_AREA and types[i][1] == UNIFORM_AREA:
                bad.append(lines[i])
        return bad, out, ins

    def find_m_n(self, line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        if x1 == x2:
            m1 = 10000000000
        else:
            m1 = float(float(y2 - y1) / float(x2 - x1))
        n1 = y1 - m1 * x1

        return m1, n1

    def gausThresholdChess(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gaus = cv2.adaptiveThreshold(img2, GAUSS_MAX_VALUE,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, GAUSS_BLOCK_SIZE,
                                     GAUSS_C)

        return gaus

    def doConv(self, img, line):
        window_length = len(img) // PartOfWindowLength
        window_height = len(img[0]) // PartOfWindowHeight

        changeCounterAbv = 0
        changeCounterBel = 0
        lastcolorFlag = -1  ##zero if last is white, 1 if black
        #    newim = copy.deepcopy(img)
        m, n = self.find_m_n(line)
        m = int(m)
        n = int(n)
        for i in range(0, len(img) - window_length - 1, window_length):
            whitepix = 0
            blackpix = 0
            for j in range(i, i + window_length, ConvSkipX):
                for k in range(m * j + n - window_height,
                               m * j + n, ConvSkipY):
                    if k >= len(img) or j >= len(img[0]):
                        break
                    if (img[k][j] > 0):
                        whitepix = whitepix + 1
                    else:
                        blackpix = blackpix + 1
            if whitepix > blackpix:
                specificcolorFlag = 1  ##zero if last is white, 1 if black
            else:
                specificcolorFlag = 0
            if lastcolorFlag == -1:
                lastcolorFlag = specificcolorFlag
            if specificcolorFlag != lastcolorFlag:
                changeCounterAbv = changeCounterAbv + 1
                lastcolorFlag = specificcolorFlag

        lastcolorFlag = -1
        for i in range(0, len(img) - window_length - 1, window_length):
            whitepix = 0
            blackpix = 0
            for j in range(i, i + window_length, ConvSkipX):
                for k in range(m * j + n,
                               m * j + n + window_height, ConvSkipY):
                    if k >= len(img) or j >= len(img[0]):
                        break
                    if (img[k][j] > 0):
                        whitepix = whitepix + 1
                    else:
                        blackpix = blackpix + 1
            if whitepix > blackpix:
                specificcolorFlag = 1  ##zero if last is white, 1 if black
            else:
                specificcolorFlag = 0
            if lastcolorFlag == -1:
                lastcolorFlag = specificcolorFlag
            if specificcolorFlag != lastcolorFlag:
                changeCounterBel = changeCounterBel + 1
                lastcolorFlag = specificcolorFlag
                # print(changeCounterAbv,changeCounterBel)
        return changeCounterAbv, changeCounterBel

    ### receives dilated img
    def is_proj_correct(self, img, line):
        # resize_image = cv2.resize(newimg, (0, 0), fx=1, fy=1)
        changeColor = self.doConv(img, line)
        if (changeColor > CHANGE_COLOR_SAF):
            return True
        return False

    def remove_bad_hor_lines(self, hor_lines, valfunc, real_img):
        newimg = self.gausThresholdChess(real_img)
        kernel = np.ones((5, 5), np.uint8)
        dilim = cv2.dilate(newimg, kernel)
        hor = sorted(hor_lines, key=lambda x: -valfunc(x))
        bottomdiff = 0
        topdiff = 0
        self.draw_lines(hor, dilim)
        for i in range(len(hor)):
            changeabv, changebel = self.doConv(dilim, hor[i])
            changediff = changeabv - changebel
            if changediff > CHANGE_DIFF:
                bottomdiff = abs(changediff)
                hor = hor[i:]
                print('found bottom line')
                print(changeabv, changebel)
                break

        self.draw_lines(hor, dilim)
        for i in range(len(hor)):
            j = len(hor) - 1 - i
            changeabv, changebel = self.doConv(dilim, hor[j])
            changediff = changeabv - changebel
            if changediff < -CHANGE_DIFF:
                topdiff = abs(changediff)
                hor = hor[:j + 1]
                print('found bottom line')
                print(changeabv, changebel)
                break
        self.draw_lines(hor, dilim)
        return hor, topdiff > bottomdiff

    """
    finds highest reasonable hor line, and returns it's index
    """

    def get_highest_horizontal_line_index(self, lines, base_idx, d,
                                          get_pos, get_theta):

        ##### Find average angle, without weird lines that are errors. #####
        tmplines = []
        baseline = lines[base_idx]

        ## remove baseline duplicates
        for line in lines:
            if not self.line_eq(baseline, line):
                tmplines.append(line)
        tmplines.append(baseline)

        ## get avg angle
        # print(tmplines)
        angles = [get_theta(l) for l in tmplines]
        angle_avg = sum(angles) / len(angles)
        angle_std = np.std(angles)
        if (DEBUG):
            print("angles:" + str(angles))
            print("angles std:" + str(angle_std))
            print("angles avg:" + str(angle_avg))
        angles2 = [angle for angle in angles if
                   abs(angle - angle_avg) < ANGLE_STD_COEFF * angle_std]
        if len(angles2) < MIN_NUM_ANGLES_AVG:
            angles2 = sorted(angles, key=lambda x: abs(x - angle_avg))
            angles2 = angles2[:MIN_NUM_ANGLES_AVG]
        angles = angles2
        if (DEBUG):
            print("after fix ")
            print("angles:" + str(angles))
            print("angles avg:" + str(angle_avg))
        angle_avg = sum(angles) / len(angles)

        ##### Find highest line that is close enough to such angle #####
        def line_metric(line, idx):
            score = (8 - idx) - LINE_METRIC_ANGLE_ERROR_COEFF * abs(get_theta(
                line) - angle_avg) - LINE_METRIC_DIST_ERROR_COEFF * self.modulo(
                abs(get_pos(line) - get_pos(baseline)), d) * 1.0 / d
            return score

        scores = [(i, line_metric(lines[i], i)) for i in range(9) if not (
            self.line_eq(lines[i], baseline) and (not i == base_idx))]
        best_line_idx = scores[(max(range(len(scores)), key=lambda x: (scores[
                                                                           x])[1]))][0]
        line_metric(lines[best_line_idx], best_line_idx)
        if (self.line_eq(lines[best_line_idx], baseline)):  # index is not
            # real. fix it by finding
            best_line_idx = base_idx

        return best_line_idx

    """
      finds highest reasonable hor line, and returns it's index
      """

    def get_best_line_pair_index(self, lines, base_idx, d,
                                 get_pos, get_theta):

        ##### Find average angle, without weird lines that are errors. #####
        tmplines = []
        baseline = lines[base_idx]

        ## remove baseline duplicates
        for line in lines:
            if not self.line_eq(baseline, line):
                tmplines.append(line)
        tmplines.append(baseline)

        ## get avg angle
        ## print(tmplines)
        angles = [get_theta(l) for l in tmplines]
        angle_avg = sum(angles) / len(angles)
        angle_std = np.std(angles)
        if (DEBUG):
            print("angles:" + str(angles))
            print("angles std:" + str(angle_std))
            print("angles avg:" + str(angle_avg))
        angles2 = [angle for angle in angles if
                   abs(angle - angle_avg) < ANGLE_STD_COEFF * angle_std]
        if len(angles2) < MIN_NUM_ANGLES_AVG:
            angles2 = sorted(angles, key=lambda x: abs(x - angle_avg))
            angles2 = angles2[:MIN_NUM_ANGLES_AVG]
        angles = angles2
        if (DEBUG):
            print("after fix ")
            print("angles:" + str(angles))
            print("angles avg:" + str(angle_avg))
        angle_avg = sum(angles) / len(angles)

        ##### Find highest line that is close enough to such angle #####
        def pair_metric(line, idx, line2, idx2):
            score = abs(idx2 - idx) - LINE_METRIC_ANGLE_ERROR_COEFF * (abs(
                get_theta(line) - angle_avg) + abs(
                get_theta(line2) - angle_avg)) - LINE_METRIC_DIST_ERROR_COEFF * (self.modulo(
                abs(get_pos(line) - get_pos(baseline)), d) + self.modulo(
                abs(get_pos(line2) - get_pos(baseline)), d)) * 1.0 / d
            return score

        scores = [(i, j, pair_metric(lines[i], i, lines[j], j)) for i in
                  range(9) if not (self.line_eq(lines[i], baseline) and (
                not i == base_idx)) for j in range(9) if (not j == i) and not (
                self.line_eq(
                    lines[j], baseline) and (
                    not j == base_idx))]
        best_pair_indices = scores[(max(range(len(scores)), key=lambda x: (
            scores[x])[2]))][0:2]
        pair_metric(lines[best_pair_indices[0]], best_pair_indices[0],
                    lines[best_pair_indices[1]], best_pair_indices[1])
        if (self.line_eq(lines[best_pair_indices[0]], baseline)):  # index is not
            # real. fix it by finding
            best_pair_indices = (base_idx, best_pair_indices[1])

        if (self.line_eq(lines[best_pair_indices[1]], baseline)):  # index
            # is not
            # real. fix it by finding
            best_pair_indices = (best_pair_indices[0], base_idx)

        return best_pair_indices

    ## convert points on real_im with spare to big_im
    def get_bigims_pts(self, pts, bigim, prevdiff, prevgrid):
        bigpts = []
        for pt in pts:
            bigpts.append([(pt[0]-RESIZE_WIDTH*prevdiff/prevgrid)*
                           prevgrid/PROJECTION_SPARE_GRID_SIZE_BIG+len(bigim[0])
                            *PROJECTION_SPARE_DIFF_BIG/PROJECTION_SPARE_GRID_SIZE_BIG,
                           (pt[1]-RESIZE_HEIGHT*prevdiff/prevgrid)*
                           prevgrid/PROJECTION_SPARE_GRID_SIZE_BIG+len(bigim)*PROJECTION_SPARE_DIFF_BIG/PROJECTION_SPARE_GRID_SIZE_BIG])

        return bigpts

    ### Cut image at different idxs and check which is best
    def get_best_cut_image(self, realim, edgeim, bigim, pts, bigpts, frame,
                           cut_spare):
        bwim = self.gausThresholdChess(realim)
        bwims = [self.projection(pts, bwim, [frame[0] + i,
                                             frame[1] + j,
                                             frame[2] + i,
                                             frame[3] + j], False,False)
                 for i in range(-BOTTOM_LINE_INDEX_VARIATION,
                                BOTTOM_LINE_INDEX_VARIATION + 1) for j
                 in range(-BOTTOM_LINE_INDEX_VARIATION,
                          BOTTOM_LINE_INDEX_VARIATION + 1)]
        mindiff = 999999999999
        miniidx = 0
        minjidx = 0
        for j in range(-BOTTOM_LINE_INDEX_VARIATION,
                       BOTTOM_LINE_INDEX_VARIATION + 1):
            for i in range(-BOTTOM_LINE_INDEX_VARIATION,
                           BOTTOM_LINE_INDEX_VARIATION + 1):
                diff = self.get_diff_area(bwims[(
                                                    i + BOTTOM_LINE_INDEX_VARIATION) * (
                                                    2 * BOTTOM_LINE_INDEX_VARIATION + 1) + (
                                                    j + BOTTOM_LINE_INDEX_VARIATION)],
                                          self.last_image_bw)
                if diff < mindiff:
                    mindiff = diff
                    miniidx = i
                    minjidx = j
        edgeim = self.projection(pts, edgeim, [frame[0] + miniidx,
                                               frame[1] + minjidx,
                                               frame[2] + miniidx,
                                               frame[3] + minjidx],
                                 cut_spare, False)
        realim = self.projection(bigpts, bigim, [frame[0] + miniidx,
                                               frame[1] + minjidx,
                                               frame[2] + miniidx,
                                               frame[3] + minjidx],
                                 cut_spare, False)
        bigim = self.projection(bigpts, bigim, [frame[0] + miniidx,
                                               frame[1] + minjidx,
                                               frame[2] + miniidx,
                                               frame[3] + minjidx],
                                 False, True)


        return realim, edgeim, bigim

    def get_diff_area(self, im1, im2):
        ctr = 0
        for i in range(0, len(im1), IM_DIFF_AREA_SKIP):
            for j in range(0, len(im1[0]), IM_DIFF_AREA_SKIP):
                if im1[i][j] != im2[i][j]:
                    ctr += 1
        return ctr

    ## image with 9th row
    def get_final_image(self, bigim):
        x_low = len(bigim[0])*PROJECTION_SPARE_DIFF_BIG/PROJECTION_SPARE_GRID_SIZE_BIG
        x_hi = len(bigim[
                        0]) * (1-PROJECTION_SPARE_DIFF_BIG / \
                               PROJECTION_SPARE_GRID_SIZE_BIG)

        y_low = len(bigim) * (PROJECTION_SPARE_DIFF_BIG-1) / PROJECTION_SPARE_GRID_SIZE_BIG
        y_hi = len(bigim) * (1 - PROJECTION_SPARE_DIFF_BIG / \
                              PROJECTION_SPARE_GRID_SIZE_BIG)

        final_img = bigim[int(y_low):int(y_hi),int(x_low):int(x_hi)]
        final_img = cv2.resize(final_img, (FINAL_IMG_WIDTH,FINAL_IMG_HEIGHT))
        return final_img

    def main(self, real_img):

        def get_theta(line):
            if line[2] == line[0]:
                return math.pi / 2
            else:
                return (float)(math.atan2(float(line[3] - line[1]),
                                          float(line[2] - line[0]))) % math.pi

            """"""

        """
         do not use with vertical lines.
         """

        def get_y_point_on_line(line):
            x1 = line[0]
            y1 = line[1]
            y = y1 - (x1 - RESIZE_WIDTH // 2) * (line[3] - y1) * 1.0 / (line[2] - x1)
            return y

        """
        do not use with horizontal lines.
        """

        def get_x_point_on_line(line):
            x1 = line[0]
            y1 = line[1]
            return x1 - (y1 - RESIZE_HEIGHT // 2) * (line[2] - x1) * 1.0 / \
                        (line[3] - y1)

        def get_theta_hor(line):
            angle = self.get_theta(line)
            if angle > math.pi / 2:
                return angle - math.pi
            return angle

        def get_theta_ver(line):
            return self.get_theta(line)

        for j in range(NUM_ANGLE_ITERATIONS):
            try:

                tmp_realimg, tmp_edgeim, tmp_bigim= self.board_id.main(real_img)

                for i in range(NUM_ITERATIONS):


                    angle_constraint = 1 * (ANGLE_CONSTRAINT_DECAY_FACTOR ** i)
                    hor, ver = self.get_lines(tmp_edgeim, angle_constraint)

                    # hor, start_from_top = self.remove_bad_hor_lines(hor,
                    #                                         get_y_point_on_line,
                    #                                  real_img)

                    # TODO: fix id of lines too low
                    new_hor, best_hor_idx, hor_d = self.get_line_series(hor,
                                                                        lambda
                                                                            x: RESIZE_HEIGHT - get_y_point_on_line(x),
                                                                        len(
                                                                            tmp_edgeim) * MIN_GRID_SIZE,
                                                                        len(
                                                                            tmp_edgeim) * MAX_GRID_SIZE, 9)
                    new_ver, best_ver_idx, ver_d = self.get_line_series(ver,
                                                                        get_x_point_on_line, len(tmp_edgeim[0]) * MIN_GRID_SIZE,
                                                                        len(
                                                                            tmp_edgeim[0]) * MAX_GRID_SIZE, 9)
                    if (DEBUG):
                        self.draw_lines(ver, tmp_edgeim)
                        self.draw_lines(new_ver, tmp_edgeim)
                        self.draw_lines(hor, tmp_edgeim)
                        self.draw_lines(new_hor, tmp_edgeim)

                    best_hor_pair = self.get_best_line_pair_index(new_hor,
                                                                  best_hor_idx, hor_d,
                                                                  get_y_point_on_line, get_theta_hor)

                    best_ver_pair = self.get_best_line_pair_index(new_ver,
                                                                  best_ver_idx,
                                                                  ver_d,
                                                                  get_x_point_on_line,
                                                                  get_theta_ver)

                    left_num = min(best_ver_pair)
                    right_num = max(best_ver_pair)
                    up_num = min(best_hor_pair)
                    down_num = max(best_hor_pair)
                    up_line = new_hor[up_num]
                    down_line = new_hor[down_num]
                    left_line = new_ver[left_num]
                    right_line = new_ver[right_num]
                    frame = [up_num, right_num, down_num, left_num]

                    points = self.get_board_limits(up_line, right_line, down_line, left_line)

                    spare = PROJECTION_SPARE_DIFF
                    grid = PROJECTION_SPARE_GRID_SIZE
                    if(i==0): #spare is from id board
                        spare = identify_board.PROJECTION_SPARE_DIFF
                        grid = identify_board.PROJECTION_SPARE_GRID_SIZE

                    bigpts = self.get_bigims_pts(points,tmp_bigim,spare,grid)

                    if (DEBUG):
                        self.draw_lines([up_line, left_line, right_line,
                                         down_line], tmp_edgeim)
                        self.draw_points(tmp_realimg, points)


                    # find where to start cutting (bottom of board)
                    tmp_realimg, tmp_edgeim, tmp_bigim= self.get_best_cut_image(
                        tmp_realimg,tmp_edgeim,tmp_bigim,
                                                               points, bigpts, frame,
                                                               i < NUM_ITERATIONS - 1)


                        # gui_img_manager.add_img(self.get_line_image(hor, edgeim))
                        # gui_img_manager.add_img(self.get_line_image(new_hor, edgeim))
                        # gui_img_manager.add_img(self.get_line_image([up_line,
                        # left_line, right_line, down_line], edgeim))
                        # gui_img_manager.add_img(proim)


                gaus = self.gausThresholdChess(tmp_realimg)
                if DEBUG:
                    cv2.imshow("image", gaus)
                    k = cv2.waitKey(0)
                is_proj_correct = board_cut_checker.board_cut_checker(gaus)

                if not is_proj_correct:
                    print("Shimri's test has failed - bad cut")
                    raise Exception()
                #return tmp_realimg
                return self.get_final_image(tmp_bigim)

            except:
                real_img = self.rotate_image_fix(real_img)
                print("rotating image")
        raise Exception()

    def get_line_image(self, lines, img):
        bin = copy.deepcopy(img)
        bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

        for i in range(0, len(lines)):
            l = lines[i]
            cv2.line(bin, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
        return bin


def test(foldername):
    # get lines from image, and edge-image
    id = identify_board.identify_board()
    fixer = board_cut_fixer()
    for j in range(0, 200):
        try:
            realim = id.get_image_from_filename(foldername + "\\" + str(j) + ".jpg")
            fixed_im = fixer.main(realim)

            cv2.imwrite(foldername + '\\fixed\\' + str(j) + '.jpg', fixed_im)
            print(str(j)+'sucsseed!!!')
        except:
            print(str(j) + " failed")


#test('angle2')
