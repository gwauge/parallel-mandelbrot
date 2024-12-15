import numpy as np
import vispy.scene
from vispy.scene import visuals

file_path = "test1"  # Replace with your file path

# Read the data into a numpy array
data = np.loadtxt(file_path, delimiter=",", dtype=np.uint8)

canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
view = canvas.central_widget.add_view()
view.camera = 'panzoom'  # Allow panning and zooming

heatmap = visuals.Image(data, cmap='viridis', parent=view.scene, texture_format="r8")

# Set aspect ratio to match the data
view.camera.set_range()
vispy.app.run()