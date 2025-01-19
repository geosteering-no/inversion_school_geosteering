import PySimpleGUI as sg

# Define the window layout
layout = [
    [sg.Text("Hello, PySimpleGUI!", key="-LABEL-", font=("Arial", 16))],
    [sg.Button("Click Me", key="-BUTTON-")],
]

# Create the window
window = sg.Window("Simple PySimpleGUI Example", layout, size=(300, 150))

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:  # Close the window
        break
    if event == "-BUTTON-":  # Button clicked
        window["-LABEL-"].update("Button Clicked!")

# Close the window
window.close()
