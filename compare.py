import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()

paths = [
    "input/images/woods1.jpg",
    "output/stereo/woods1_right.png"
    # "output/images/dog1_depth.png",
    # "output/images/dog1_depth_2.png"
]

images = [ImageTk.PhotoImage(Image.open(p)) for p in paths]
current = 0

label = tk.Label(root, image=images[0])
label.pack()

path_label = tk.Label(root, text=paths[0])
path_label.pack()

def toggle(e=None):
    global current
    current ^= 1
    label.config(image=images[current])
    path_label.config(text=paths[current])

root.bind("<Button-1>", toggle)

root.mainloop()