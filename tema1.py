import itertools
import cv2 as cv
import numpy as np
import os

X = 720
Y = 1170
W = 1765
H = 1770

CELL_W = 60
CELL_H = 60

BOARD_W = 900
BOARD_H = 900

MINUS_INF = -np.inf
PATH_READ = 'antrenare/'
PATH_WRITE = 'rezolvare/antrenare/'
PATH_TEMPLATES = 'templates/'

player_positions = [0, 0]

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.7,fy=0.7)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_board(image):
    image_BGR = image.copy()
    cropped_image_BGR = image_BGR[Y:Y+H, X:X+W]
    
    image_m_blur = cv.medianBlur(image,9)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)

    image_HSV = cv.cvtColor(image_sharpened, cv.COLOR_BGR2HSV)
    cropped_image_HSV = image_HSV[Y:Y+H, X:X+W]
    
    low_HSV = (50, 0, 0)
    high_HSV = (120, 255, 255)
    mask_HSV = cv.inRange(cropped_image_HSV, low_HSV, high_HSV)

    kernel = np.ones((9, 9), np.uint8)
    open_image = cv.morphologyEx(mask_HSV, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(open_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    c = max(contours, key=cv.contourArea)

    xmin = min(c[:, :, 0])[0]
    ymin = min(c[:, :, 1])[0]
    xmax = max(c[:, :, 0])[0]
    ymax = max(c[:, :, 1])[0]

    topLeft = xmin, ymin
    topRight = xmax, ymin
    bottomLeft = xmin, ymax
    bottomRight = xmax, ymax

    width = BOARD_W
    height = BOARD_H

    source = np.array([topLeft,topRight,bottomRight,bottomLeft], dtype = "float32")
    destination = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(source, destination)

    result = cv.warpPerspective(cropped_image_BGR, M, (width, height))
    
    return result

def compute_combinations(N):
    values = [i for i in range(N)]
    combinations = itertools.combinations(values, 2)
    combinations = list(combinations)
    for i in range(N):
        combinations.append((i, i))
    combinations.sort(key=lambda tuple: (tuple[0], tuple[1])) 
    return combinations

def classify_domino(patch_coords, thresh):
    maxi=MINUS_INF
    poz=-1
    
    combinations = compute_combinations(7)
    N = len(combinations)

    templates = list()

    x_min, x_max, y_min, y_max, _ = patch_coords
    patch = thresh[x_min:x_max, y_min:y_max]

    for j in range(N):
        str = PATH_TEMPLATES + f"{combinations[j][0]}_{combinations[j][1]}.jpg"

        template = cv.imread(str)
        template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)
        _, template = cv.threshold(template, 30, 255, cv.THRESH_BINARY)

        template_vertical1 = template.copy()
        templates.append(template_vertical1)

        template_vertical2 = cv.rotate(template, cv.ROTATE_180)
        templates.append(template_vertical2)

        template_horizontal1 = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)
        templates.append(template_horizontal1)

        template_horizontal2 = cv.rotate(template, cv.ROTATE_90_COUNTERCLOCKWISE)
        templates.append(template_horizontal2)
    
    matches = []

    image_is_vertical = patch.shape[0] > patch.shape[1]
    for j in range(4*N):
        if (image_is_vertical == True and j%4 < 2) or (image_is_vertical == False and j%4 >= 2):
            match = cv.matchTemplate(patch, templates[j], cv.TM_CCOEFF_NORMED)
            matches.append((np.max(match), j%4))

    for j in range(2*N):
        if matches[j][0]>maxi:
            maxi=matches[j][0]
            poz=int(j/2)
            template_position = matches[j][1]

    if template_position == 0 or template_position == 3:
        return (combinations[poz][0], combinations[poz][1], maxi)

    return (combinations[poz][1], combinations[poz][0], maxi)

def get_domino_matrix(thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype=np.int32)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_mean=np.mean(patch)
            if patch_mean>=100:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return matrix

def get_patches_from_matrix(matrix, lines_horizontal, lines_vertical):
    patches = list()
    patches_coords = list()
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            if(matrix[i][j] == 1):
                y_min = lines_vertical[j][0][0]
                y_max = lines_vertical[j + 1][1][0]
                x_min = lines_horizontal[i][0][1]
                x_max = lines_horizontal[i + 1][1][1]
                topLeft = (x_min, y_min)
                topRight = (x_max, y_min)
                bottomRight = (x_max, y_max)
                bottomLeft = (x_min, y_max)
                patches_coords.append((topLeft, topRight, bottomRight, bottomLeft))
    
    for i in range(len(patches_coords)):
        for j in range(i+1, len(patches_coords)):
            orientation = None
            topLeft = (-1, -1)
            bottomRight = (-1, -1)
            topRight = (-1, -1)
            bottomLeft = (-1, -1)

            topLeft_crt, topRight_crt, bottomRight_crt, bottomLeft_crt = patches_coords[i]
            topLeft_next, topRight_next, bottomRight_next, bottomLeft_next = patches_coords[j]

            # vertical
            if ((topLeft_crt == bottomLeft_next and topRight_crt == bottomRight_next) or
                (topLeft_next == bottomLeft_crt and topRight_next == bottomRight_crt)):
                    topLeft = min(topLeft_crt, topLeft_next, key=lambda tuple: tuple[1])
                    topRight = min(topRight_crt, topRight_next, key=lambda tuple: tuple[1])
                    bottomLeft = max(bottomLeft_crt, bottomLeft_next, key=lambda tuple: tuple[1])
                    bottomRight = max(bottomRight_crt, bottomRight_next, key=lambda tuple: tuple[1])
                    orientation = 'v'

            # orizontal
            elif ((topLeft_crt == topRight_next and bottomLeft_crt == bottomRight_next) or
                (topLeft_next == topRight_crt and bottomLeft_next == bottomRight_crt)):
                    topLeft = min(topLeft_crt, topLeft_next, key=lambda tuple: tuple[0])
                    bottomLeft = min(bottomLeft_crt, bottomLeft_next, key=lambda tuple: tuple[0])
                    topRight = max(topRight_crt, topRight_next, key=lambda tuple: tuple[0])
                    bottomRight = max(bottomRight_crt, bottomRight_next, key=lambda tuple: tuple[0])
                    orientation = 'h'

            x_min, y_min = topLeft
            x_max, y_max = bottomRight

            if x_min != -1 and x_max != -1 and y_min != -1 and y_max != -1:
                patches.append((x_min, x_max, y_min, y_max, orientation))

    return patches

def calculate_score(crt_domino, crt_player):
    player_points = [0, 0]

    diamonds = [[(4, 4), (6, 4), (5, 5), (4, 6),
                (4, 8), (4, 10), (5, 9), (6, 10),
                (8, 4), (10, 4), (9, 5), (10, 6),
                (8, 10), (9, 9), (10, 10), (10, 8)],
                [(2, 4), (3, 5), (4, 2), (5, 3), 
                 (2, 10), (3, 9), (4, 12), (5, 11),  
                 (9, 3), (10, 2), (11, 5), (12, 4),
                 (9, 11), (10, 12), (11, 9), (12, 10)],
                [(7, 0), (7, 14), (0, 7), (14, 7),
                 (2, 1), (1, 2), (3, 3), (1, 12),
                 (2, 13), (3, 11), (11, 3), (12, 1),
                 (13, 2), (11, 11), (12, 13), (13, 12)],
                [(3, 0), (0, 3), (1, 5), (5, 1),
                 (0, 11), (1, 9), (3, 14), (5, 13),
                 (11, 0), (9, 1), (14, 3), (13, 5),
                 (9, 13), (11, 14), (13, 9), (14, 11)],
                [(0, 0), (14, 0), (14, 14), (0, 14)]]

    points_per_position = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 
                           4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 
                           1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2, 5, 
                           0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5, 6, 
                           2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]
    
    (i1, j1, value1), (i2, j2, value2) = crt_domino
    
    for idx, list in enumerate(diamonds):
        for diamond in list:
            if ((i1, j1) == diamond or (i2, j2) == diamond) and (value1 == value2):
                player_points[crt_player] += 2 * (idx + 1)
            elif (i1, j1) == diamond or (i2, j2) == diamond:
                player_points[crt_player] += idx + 1

    if points_per_position[player_positions[0]] == value1 or points_per_position[player_positions[0]] == value2:
        player_points[0] += 3
    
    if points_per_position[player_positions[1]] == value1 or points_per_position[player_positions[1]] == value2:
        player_points[1] += 3

    return player_points

def main():
    lines_horizontal=[]
    for i in range(0,BOARD_H+1,CELL_H):
        l=[]
        l.append((0,i))
        l.append((BOARD_H-1,i))
        lines_horizontal.append(l)

    lines_vertical=[]
    for i in range(0,BOARD_W+1,CELL_W):
        l=[]
        l.append((i,0))
        l.append((i,BOARD_W-1))
        lines_vertical.append(l)

    games = []
    game_idx = -1
    idx = -1
    files=os.listdir(PATH_READ)
    files.sort()
    cols = [chr(ord('A') + k) for k in range(15)]
    
    for file in files:
        if file[1:] == '_mutari.txt':
            move_idx = 0
            game_moves = {}
            with open(PATH_READ+file) as f:
                for line in f:
                    move_idx += 1
                    if 'player1' in line:
                        game_moves[move_idx] = 0
                    else:
                        game_moves[move_idx] = 1
            games.append(game_moves)
    
    for file in files:     
        if file[-3:]=='jpg':
            idx += 1
            if file[1:4] == '_01':
                game_idx += 1

                global player_positions
                player_positions = [0, 0]

                matrix_prev = np.zeros((15, 15), dtype=np.int32)
            else:
                matrix_prev = np.copy(matrix_crt)

            img = cv.imread(PATH_READ+file)

            board=get_board(img)

            board_GRAY = cv.cvtColor(board,cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(board_GRAY, 185, 255, cv.THRESH_BINARY)

            matrix_crt = get_domino_matrix(thresh, lines_horizontal, lines_vertical)
            patches = get_patches_from_matrix(matrix_crt-matrix_prev, lines_horizontal, lines_vertical)

            maxi = MINUS_INF
            domino_values = (-1, -1)
            final_patch_coords = (-1, -1, -1, -1)

            for patch in patches:
                res = classify_domino(patch, thresh)
                if res[2] > maxi:
                    maxi = res[2]
                    domino_values = res[0], res[1]
                    final_patch_coords = patch
            
            x_min, _, y_min, _, orientation = final_patch_coords
            
            i1 = int(x_min / CELL_W)
            j1 = int(y_min / CELL_H)
            
            if(orientation == 'v'):
                i2 = i1
                j2 = j1 + 1
            elif(orientation == 'h'):
                i2 = i1 + 1
                j2 = j1

            matrix_crt = np.copy(matrix_prev)
            matrix_crt[i1][j1] = 1
            matrix_crt[i2][j2] = 1

            crt_domino = ((i1, j1, domino_values[0]), (i2, j2, domino_values[1]))
            crt_player = games[game_idx][idx%20+1]
            player_points = calculate_score(crt_domino, crt_player)

            player_positions[0] += player_points[0]
            player_positions[1] += player_points[1]

            with open(PATH_WRITE + f'{file[:4]}.txt', 'w') as f:
                f.write(f"{i1+1}{cols[j1]} {domino_values[0]}\n")
                f.write(f"{i2+1}{cols[j2]} {domino_values[1]}\n")
                f.write(f"{player_points[crt_player]}\n")

            print(f"Am scris {file[:4]}.txt")

main()
