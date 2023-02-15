# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
# culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
# culane_row_anchor = [106, 117, 127, 138, 149, 159, 170, 181, 191, 202, 213, 223, 234, 245, 255, 266, 277, 287]
# culane_row_anchor = [66, 79, 92, 105, 118, 131, 144, 157, 170, 183, 196, 209, 222, 235, 248, 261, 274, 287]
# culane_row_anchor = [49, 63, 77, 91, 105, 119, 133, 147, 161, 175, 189, 203, 217, 231, 245, 259, 273, 287]
culane_row_anchor = [32, 47, 62, 77, 92, 107, 122, 137, 152, 167, 182, 197, 212, 227, 242, 257, 272, 287]

