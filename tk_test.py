import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter Example")
root.geometry("300x200")  # Set the size of the window

# Create a label
label = tk.Label(root, text="Hello, Tkinter!", font=("Arial", 16))
label.pack(pady=20)  # Add some padding around the label

# Create a function for the button
def on_button_click():
    label.config(text="Button Clicked!")

# Create a button
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack(pady=10)

# Run the main event loop
root.mainloop()

