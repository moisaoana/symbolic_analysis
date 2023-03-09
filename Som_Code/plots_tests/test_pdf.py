from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Define the page size
PAGE_WIDTH, PAGE_HEIGHT = A4

# Create a new canvas
c = canvas.Canvas("../images.pdf", pagesize=A4)

# Define the top and bottom margins
TOP_MARGIN = 0
BOTTOM_MARGIN = 0
img_height = 280

# Define the starting y-coordinate for the first image
y = PAGE_HEIGHT - TOP_MARGIN - img_height

# List of image paths
images = ['test_image.png', 'test_image.png', 'test_image.png', 'test_image.png', 'test_image.png']

# Loop through the images and draw them on the canvas
for img in images:
    # Check if there is enough space on the current page for the image
    if y >= BOTTOM_MARGIN:
        # Draw the image on the current page
        print("same page")
        c.drawImage(ImageReader(img), 10, y, width=600, height=280, preserveAspectRatio=True)

        # Update the y-coordinate for the next image
        y -= img_height
    else:
        # Create a new page and draw the image on the new page
        print("new page")
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN - img_height
        c.drawImage(ImageReader(img), 10, y, width=600, height=280, preserveAspectRatio=True)
        y -= img_height

# Save the canvas as a PDF file
c.save()
