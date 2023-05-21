import numpy as np
import cv2
import sys
sys.setrecursionlimit(10000000)

img = cv2.imread("../resources/assets/img1.png")


def preprocess(img):

    lower = np.array([143, 82, 47])
    upper = np.array([143, 82, 47])
    mask = cv2.inRange(img, lower, upper)
    outer_box = cv2.bitwise_and(img, img, mask=mask)

    lower = np.array([196, 114, 68])
    upper = np.array([196, 114, 68])
    mask = cv2.inRange(img, lower, upper)
    points = cv2.bitwise_and(img, img, mask=mask)
    cords = cv2.findNonZero(mask)

    start = cords[0][0]
    end = []
    for cord in cords:
        if (cord[0][1] - start[1] >= 80):
            end = cord[0]
            break

    lower = np.array([249, 249, 249])  # -- Lower range --
    upper = np.array([255, 255, 255])  # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    secondary_roads = cv2.bitwise_and(img, img, mask=mask)

    lower = np.array([140, 140, 140])  # -- Lower range --
    upper = np.array([190, 190, 190])  # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    primary_roads = cv2.bitwise_and(img, img, mask=mask)

    res = cv2.bitwise_or(primary_roads, secondary_roads)

    res = cv2.bitwise_or(res, outer_box)
    res = cv2.bitwise_or(res, points)
    return res, start, end


res, start, end = preprocess(img)
grid_shape = (60, 150)
h, w, _ = img.shape
rows, cols = grid_shape
dy, dx = h / rows, w / cols
x_grid = [int(round(x)) for x in np.linspace(start=0, stop=w, num=cols-1)]
y_grid = [int(round(y)) for y in np.linspace(start=0, stop=h, num=rows-1)]
cell_cords = []


for x_idx, x in enumerate(x_grid[:-1]):
    buf_list = []
    for y_idx, y in enumerate(y_grid[:-1]):
        buf_list.append(((x_grid[x_idx], y_grid[y_idx]),
                        (x_grid[x_idx + 1], y_grid[y_idx + 1])))
    cell_cords.append(buf_list)


def convert_px_to_cell(px):
    x_px, y_px = px[0], px[1]
    for idx_x, cell_x in enumerate(cell_cords):
        for idx_y, cell_data in enumerate(cell_x):
            if (x_px >= cell_data[0][0] and x_px <= cell_data[1][0]):
                if (y_px >= cell_data[0][1] and y_px <= cell_data[1][1]):
                    return (idx_x, idx_y)


cell1 = convert_px_to_cell(start)
cell2 = convert_px_to_cell(end)
start_cell = None
end_cell = None
if cell1[0] > cell2[0]:
    start_cell = cell2
    end_cell = cell1
else:
    start_cell = cell1
    end_cell = cell2


def draw_grid(img, color=(0, 255, 0), thickness=1):

    for x in x_grid:
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    for y in y_grid:
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


def get_grid_cell(img, cord):
    y1, y2 = 0, 0
    x1, x2 = 0, 0

    for y in np.linspace(start=0, stop=h-dy, num=rows-1):
        y = int(round(y))
        if y >= cord[1]:
            y2 = int(round(y))
            y1 = int(round(y2) - round(dy))
            break

    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        if x >= cord[0]:
            x2 = int(round(x))
            x1 = int(round(x2) - round(dx))
            break

    if x1 <= dx:
        x1 = 0
    if y1 <= dx:
        y1 = 0

    crop = img[x1:x2, y1:y2]
    return crop, x1, y1, x2, y2


def fill_cell(img, cord, color=(0, 0, 255)):

    x1, y1 = cord[0]
    x2, y2 = cord[1]
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    img = cv2.rectangle(img, pt1, pt2, color, thickness=-1)

    return img


def invert(img):
    return (255-img)


def encode_gen(gen):
    gen_buf = gen
    gen1 = gen_buf // (255*255)
    gen_buf = gen_buf - ((255*255) * gen1)
    gen2 = gen_buf // (255)
    gen3 = gen_buf - (gen2 * 255)

    return gen1, gen2, gen3


def decode_gen(gen1, gen2, gen3):
    buf = 0
    buf += (gen1 * (255*255)) + (gen2 * 255) + gen3
    return buf


def delete_dots(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1]
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 6:
            img2[labels == i + 1] = 255

    res = cv2.bitwise_not(img2)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)


reference_image = res.copy()
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_image = delete_dots(reference_image)
reference_image = invert(reference_image)
reference_image_bw = cv2.cvtColor(reference_image.copy(), cv2.COLOR_BGR2GRAY)
change_pic = img.copy()
checked_cords = []


def crop(cell):
    ref = reference_image_bw.copy()
    return ref[cell[0][1]:cell[1][1], cell[0][0]:cell[1][0]]


route = []


def recursive_alg(cell_idx, gen=-1, call_from='start'):
    gen += 1
    x, y = cell_idx[0], cell_idx[1]
    for cord in checked_cords:
        if cell_cords[x][y] == cord:
            return

    checked_cords.append(cell_cords[x][y])

    if (cell_cords[x][y] == cell_cords[end_cell[0]][end_cell[1]]):
        fill_cell(reference_image, cell_cords[x][y], encode_gen(gen))
        route.append((x, y))
        return True

    if cv2.findNonZero(crop(cell_cords[x][y])) is not None:
        fill_cell(reference_image, cell_cords[x][y], encode_gen(gen))
        fill_cell(change_pic, cell_cords[x][y], encode_gen(gen))
    else:
        return

    if call_from == 'start':
        # down
        if (y != len(y_grid) - 2):
            if recursive_alg((x, y + 1), gen=gen, call_from='down'):
                return True

    if call_from == 'left':
        # up
        if (y != 0):
            if recursive_alg((x, y - 1), gen=gen, call_from='up'):
                return True
        # left
        if (x != 0):
            if recursive_alg((x - 1, y), gen=gen, call_from='left'):
                return True
        # down
        if (y != len(y_grid) - 2):
            if recursive_alg((x, y + 1), gen=gen, call_from='down'):
                return True

    if call_from == 'down':
        # left
        if (x != 0):
            if recursive_alg((x - 1, y), gen=gen, call_from='left'):
                return True
        # right
        if (x != len(x_grid) - 2):
            if recursive_alg((x + 1, y), gen=gen, call_from='right'):
                return True
        # down
        if (y != len(y_grid) - 2):
            if recursive_alg((x, y + 1), gen=gen, call_from='down'):
                return True

    if call_from == 'right':
        # up
        if (y != 0):
            if recursive_alg((x, y - 1), gen=gen, call_from='up'):
                return True
        # right
        if (x != len(x_grid) - 2):
            if recursive_alg((x + 1, y), gen=gen, call_from='right'):
                return True
        # down
        if (y != len(y_grid) - 2):
            if recursive_alg((x, y + 1), gen=gen, call_from='down'):
                return True

    if call_from == 'up':
        # up
        if (y != 0):
            if recursive_alg((x, y - 1), gen=gen, call_from='up'):
                return True
        # right
        if (x != len(x_grid) - 2):
            if recursive_alg((x + 1, y), gen=gen, call_from='right'):
                return True
        # left
        if (x != len(x_grid) - 2):
            if recursive_alg((x - 1, y), gen=gen, call_from='right'):
                return True
        # down
        if (y != len(y_grid) - 2):
            if recursive_alg((x, y + 1), gen=gen, call_from='down'):
                return True


def go_back(cords, previos=(123123, 123123)):

    def get_cell_code(cords):
        x_cell, y_cell = cords[0], cords[1]
        start, end = cell_cords[x_cell][y_cell]
        gen1 = ref[start[1], start[0]][0]
        gen2 = ref[start[1], start[0]][1]
        gen3 = ref[start[1], start[0]][2]
        return decode_gen(gen1, gen2, gen3)

    def check_on_start(x, y):
        if (cell_cords[x][y] == cell_cords[start_cell[0]][start_cell[1]]):
            route.append((x, y))
            return True
        return False

    base_flag_code = get_cell_code(cords)
    min_flag_code = base_flag_code
    way = None
    x, y = cords[0], cords[1]

    for i in range(0, 4):
        x -= i
        if x >= 0:
            if check_on_start(x, y):
                return True
            flag = get_cell_code((x, y))
            if flag != 0 and flag <= min_flag_code:
                min_flag_code = flag
                way = (x, y)

        x += i

        x += i
        if (x <= len(x_grid) - 2):
            if check_on_start(x, y):
                return True
            flag = get_cell_code((x, y))
            if flag != 0 and flag <= min_flag_code:
                min_flag_code = flag
                way = (x, y)

        x -= i

        y += i
        if (y <= len(y_grid) - 2):
            if check_on_start(x, y):
                return True
            flag = get_cell_code((x, y))
            if flag != 0 and flag <= min_flag_code:
                min_flag_code = flag
                way = (x, y)
        y -= i

        y -= i
        if (y >= 0):
            if check_on_start(x, y):
                return True
            flag = get_cell_code((x, y))
            if flag != 0 and flag <= min_flag_code:
                min_flag_code = flag
                way = (x, y)
        y += i

    if way is None:
        raise ValueError("Can't find a way to start")
    # if way == previos:
    #     way = (way[0], way[1] - 1)
    #     previos = way

    if previos == way:
        way = (way[0] - 2, way[1] + 2)

    route.append((way))
    previos = way
    go_back(way, previos=previos)


recursive_alg(start_cell)
ref = reference_image.copy()
go_back(end_cell)

result_image = img.copy()
for r in route:
    result_image = fill_cell(result_image, cell_cords[r[0]][r[1]])

reference_image = draw_grid(reference_image)
cv2.imshow('edge', reference_image)
cv2.imshow('edge1', reference_image_bw)
cv2.imshow('res', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
