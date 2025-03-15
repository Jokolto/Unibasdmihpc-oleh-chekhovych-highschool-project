import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from numba import njit, prange
from numba.openmp import openmp_context as omp
from numba.openmp import omp_get_num_threads, omp_set_num_threads, omp_get_thread_num

def mandelbrot_calculate(z, p, c):
    return z**p + c

# Predefined 'nipy_spectral' colormap (RGB values for a normalized [0, 1] range)
nipy_spectral_colormap = [
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.6), (0.0, 0.0, 0.8), (0.0, 0.4, 0.9), 
    (0.2, 0.6, 0.9), (0.4, 0.8, 0.8), (0.6, 1.0, 0.6), (0.8, 1.0, 0.2), 
    (1.0, 1.0, 0.0), (1.0, 0.8, 0.0), (1.0, 0.6, 0.0), (1.0, 0.4, 0.0), 
    (1.0, 0.2, 0.0), (0.8, 0.0, 0.0), (0.6, 0.0, 0.0), (0.4, 0.0, 0.0)
]

# Normalize colormap values into a range of 0 to 1
def nipy_spectral_colormap_fn(value):
    # Clip value between 0 and 1
    value = np.clip(value, 0, 1)
    
    # Find the appropriate index in the colormap
    index = int(value * (len(nipy_spectral_colormap) - 1))
    return np.array(nipy_spectral_colormap[index])

# Custom colormap function: maps iteration count to RGB color using nipy_spectral
def apply_colormap(iteration_array, max_iterations):
    # Create an empty RGB image
    color_image = np.zeros((iteration_array.shape[0], iteration_array.shape[1], 3), dtype=np.uint8)

    # Normalize the iteration values to [0, 1] range
    norm_array = iteration_array / max_iterations

    # Apply the 'nipy_spectral' colormap
    for i in range(iteration_array.shape[0]):
        for j in range(iteration_array.shape[1]):
            norm_val = norm_array[i, j]
            # Map the normalized value to an RGB color using the colormap function
            color_image[i, j] = (nipy_spectral_colormap_fn(norm_val) * 255).astype(np.uint8)

    return color_image

# computing 2-d array to represent the mandelbrot-set
@njit
def mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= 2:
            return iterationNumber
        z = z**2 + c

    return 0


@njit
def compute_points(xDomain, yDomain, max_iterations, iterationArray, use_omp, num_threads):
    if use_omp:
        # work = {i: 0 for i in range(num_threads)}
        omp_set_num_threads(num_threads)
        with omp('parallel'):
            with omp('for schedule(static, 1)'):
                for y_i in range(len(yDomain)):
                    for x_i in range(len(xDomain)):
                        c = complex(xDomain[x_i], yDomain[y_i])
                        z = mandelbrot_pixel(c, max_iterations)
                        # work[omp_get_thread_num()] += z
                        iterationArray[y_i, x_i] = z
    else:
        # work = {}
        for y_i in range(len(yDomain)):
            for x_i in range(len(xDomain)):
                c = complex(xDomain[x_i], yDomain[y_i])
                iterationArray[y_i, x_i] = mandelbrot_pixel(c, max_iterations)
    return iterationArray


class MandelbrotViewer:
    def __init__(self, window, label):
        self.window = window
        self.label = label
        self.use_omp = True
        self.num_threads = 12
        # Initial viewport coordinates for the Mandelbrot set
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5

        # Number of points and max iterations
        self.num_points = 1000  # Fixed number of points (low resolution for computation)
        self.max_iterations = 100

        # Initial zoom factor
        self.zoom_factor = 1.0

        # Set the desired display resolution (higher resolution display)
        self.display_width = 800  # Width of the displayed image
        self.display_height = 800  # Height of the displayed image

        # Stretching settings 
        self.stretching_enabled = False  # Change to False if stretching is not needed
        self.stretching_factor = 1.5  # Factor by which the image is stretched (can be adjusted)

        # Bind mouse scroll for zoom
        self.window.bind("<MouseWheel>", self.zoom)

        # Initialize dragging state
        self.is_dragging = False
        self.prev_x = 0
        self.prev_y = 0

        # Bind mouse press and release for panning
        self.window.bind("<ButtonPress-1>", self.start_drag)
        self.window.bind("<B1-Motion>", self.drag)
        self.window.bind("<ButtonRelease-1>", self.end_drag)

        # Start the image update loop
        self.update_image()

    def zoom(self, event):
        # Zoom in or out depending on the mouse wheel direction
        zoom_speed = 0.1
        if event.delta > 0:  # Zoom in
            self.zoom_factor *= (1 + zoom_speed)
        else:  # Zoom out
            self.zoom_factor *= (1 - zoom_speed)

        # Update the viewport based on the zoom factor
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        # Apply the zoom factor to the viewport
        if event.delta > 0:  # Zoom in
            self.xmin += width * zoom_speed
            self.xmax -= width * zoom_speed
            self.ymin += height * zoom_speed
            self.ymax -= height * zoom_speed
        else:  # Zoom out
            self.xmin -= width * zoom_speed
            self.xmax += width * zoom_speed
            self.ymin -= height * zoom_speed
            self.ymax += height * zoom_speed

        # Recompute the image with the new zoom level
        self.update_image()

    def start_drag(self, event):
        # Mark the start of the drag
        self.is_dragging = True
        self.prev_x = event.x
        self.prev_y = event.y

    def drag(self, event):
        # Calculate how far the mouse has moved
        dx = event.x - self.prev_x
        dy = event.y - self.prev_y

        # Adjust the complex plane based on mouse movement
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        # Move the viewport by a fraction of the mouse movement
        self.xmin -= dx * width / self.window.winfo_width()
        self.xmax -= dx * width / self.window.winfo_width()

        # Invert the direction of y-axis to match the dragging behavior
        self.ymin -= dy * height / self.window.winfo_height()  # Correcting y-axis inversion
        self.ymax -= dy * height / self.window.winfo_height()

        # Update the previous mouse position for the next drag event
        self.prev_x = event.x
        self.prev_y = event.y

        # Recompute the image after panning
        self.update_image()

    def end_drag(self, event):
        # End the dragging
        self.is_dragging = False

    def update_image(self):
        # Compute Mandelbrot set as a numpy array
        xDomain, yDomain = np.linspace(self.xmin, self.xmax, self.num_points), np.linspace(self.ymin, self.ymax, self.num_points)
        iterationArray = np.zeros((self.num_points, self.num_points), dtype=int)
        if not self.use_omp:
            self.num_threads = 1
        arguments = (xDomain, yDomain, self.max_iterations, iterationArray, self.use_omp, self.num_threads)
        img_array = compute_points(*arguments)

        # Apply the custom 'nipy_spectral' colormap
        colored_img = apply_colormap(img_array, max_iterations=self.max_iterations)

        # Convert the numpy array to a PIL Image
        pil_img = Image.fromarray(colored_img)

        # Stretch or not stretch depending on the setting
        if self.stretching_enabled:
            pil_img_resized = pil_img.resize(
                (int(self.display_width), int(self.display_height)),
                Image.Resampling.LANCZOS
            )
        else:
            pil_img_resized = pil_img

        # Convert the resized PIL image to a Tkinter PhotoImage object
        img_tk = ImageTk.PhotoImage(pil_img_resized)

        # Update the image in the label
        self.label.config(image=img_tk)
        self.label.image = img_tk  # Keep a reference to the image to prevent garbage collection

# Create the Tkinter window
window = tk.Tk()
window.title("Mandelbrot Set")

# Create a label to display the image
label = tk.Label(window)
label.pack()

# Create the Mandelbrot viewer with zooming and panning
viewer = MandelbrotViewer(window, label)

# Start the Tkinter event loop
window.mainloop()
