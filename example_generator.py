from graphics import GraphWin

win = GraphWin(title= 'board', width = 16,height = 8)
canvas = win.canvas()

offset_x = 10      # Distance from left edge.
offset_y = 10      # Distance from top.
cell_size = 10     # Height and width of checkerboard squares.

points = {(1,1)}
example = []
for i in range(16):             # Note that i ranges from 0 through 7, inclusive.
    for j in range(8):
        if (i,j) in points:
            canvas.setFill('black')
            example.append(1)
        else:
            canvas.setFill('white')
            example.append(0)
        canvas.drawRect(offset_x + i * cell_size, offset_y + j * cell_size,
                        cell_size, cell_size)


example = [
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0]
]

print(example)
