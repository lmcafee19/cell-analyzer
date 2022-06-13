# This is a sample Python script.
import cv2
import tkinter
import tkinter.messagebox
import ctypes
from tkinter import *

# Get Info about user's screen size
USER = ctypes.windll.user32
SCREEN_SIZE = (USER.GetSystemMetrics(0), USER.GetSystemMetrics(1), USER.GetSystemMetrics(2), USER.GetSystemMetrics(3))


# Create Class to handle window
class Win(Tk):
    # Construct Window object
    def __init__(self, ws):
        Tk.__init__(self, ws)
        self.ws = ws
        # Set title
        self.title = "Cell Analyzer"
        # Set window size to half the screen size
        self.geometry("%sx%s" % (str(int(SCREEN_SIZE[0] / 2)), str(int(SCREEN_SIZE[1] / 2))))
        self.main()

    # Pack Default Widgets Here
    def main(self):
        print('Window Created')
        # self.w = Frame1(self)
        # self.w.pack()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("Your OpenCV version is: " + cv2.__version__)
    print("Your tkinter version is: " + str(tkinter.TkVersion))

# Sample function for button which displays a welcome message
def greet():
    tkinter.messagebox.showinfo("Greetings", "Hello! Welcome to Cell Analyzer.")

if __name__ == '__main__':
    print_hi('PyCharm')

    # Testing creating a window with tkinter
    win = Win(None)  # creating the main window and storing the window object in 'win'

    # Create form using frames
    # Create Frame. Frames are containers which can hold other widgets and place them
    frame1 = Frame(win, padx=5, pady=5)
    frame1.grid(row=0, column=1)

    Label(frame1, text='Name', padx=5, pady=5).pack()
    Label(frame1, text='Email', padx=5, pady=5).pack()
    Label(frame1, text='Password', padx=5, pady=5).pack()

    frame2 = Frame(win, padx=15, pady=15)
    frame2.grid(row=0, column=2)

    Entry(frame2).pack(padx=5, pady=5)
    Entry(frame2).pack(padx=5, pady=5)
    Entry(frame2).pack(padx=5, pady=5)

    # Create button which calls func
    Button(win, text='Submit', padx=10, command=greet).grid(row=1, columnspan=5, pady=5)

    # Create Selectable List
    frame3 = Frame(win, padx=5, pady=5)
    frame3.grid(row=0, column=3)
    lb = Listbox(frame3)
    lb.insert(1, 'Soda')
    lb.insert(2, 'Cider')
    lb.insert(3, 'Sparkling Water')
    lb.insert(4, 'Coffee')
    lb.insert(5, 'Tea')
    lb.insert(6, 'Others')
    lb.pack()



    win.mainloop()  # running the loop that works as a trigger


