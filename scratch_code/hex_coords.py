import math
import numpy as np


def calc_polygons_new(startx, starty, endx, endy, radius):
    sl = (2 * radius) * math.tan(math.pi / 6)
    print "sl: ", sl
    
    # calculate coordinates of the hexagon points
    p = sl * 0.5
    b = sl * math.cos(math.radians(30))
    w = b * 2
    h = 2 * sl
    
    print p, b, w, h


    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    row = 1

    shifted_xs = []
    straight_xs = []
    shifted_ys = []
    straight_ys = []

    while startx < endx:
        xs = [startx, startx, startx + b, startx + w, startx + w, startx + b, startx]
        straight_xs.append(xs)
        shifted_xs.append([xoffset + x for x in xs])
        startx += w

    while starty < endy:
        ys = [starty + p, starty + (3 * p), starty + h, starty + (3 * p), starty + p, starty, starty + p]
        (straight_ys if row % 2 else shifted_ys).append(ys)
        starty += yoffset
        row += 1

    polygons = [zip(xs, ys) for xs in shifted_xs for ys in shifted_ys] + [zip(xs, ys) for xs in straight_xs for ys in straight_ys]
    return polygons
    
    
startx = 0
starty = 0

endx = 5
endy = 3

radius = 1

polys = calc_polygons_new(startx, starty, endx, endy, radius)

print polys


>>> b = np.zeros((3,5,2))
>>> h = 1
>>> w = math.sqrt(3)* h / 2.0

>>> for i in range(1, rows+1):
...     for j in range(1, cols+1):
...             if i%2 == 0:
...                     b[i-1,j-1] = np.array([i*w,(j-1/2.)*h])
...             else:
...                     b[i-1,j-1] = np.array([i*w,(j)*h])
...
>>> b
array([[[ 0.8660254 ,  1.        ],
        [ 0.8660254 ,  2.        ],
        [ 0.8660254 ,  3.        ]],

       [[ 1.73205081,  0.5       ],
        [ 1.73205081,  1.5       ],
        [ 1.73205081,  2.5       ]],

       [[ 2.59807621,  1.        ],
        [ 2.59807621,  2.        ],
        [ 2.59807621,  3.        ]]])
>>> math.sqrt((b[0,1,0]-b[0,0,0])**2 + (b[0,1,1] - b[0,0,1])**2)
1.0
>>> math.sqrt((b[1,0,0]-b[0,0,0])**2 + (b[1,0,1] - b[0,0,1])**2)
0.9999999999999999
>>>

# discover radius and hexagon
apothem = .9 * (xpix[1] - xpix[0]) / math.sqrt(3)
area_inner_circle = math.pi * (apothem ** 2)














