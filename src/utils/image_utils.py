from PIL import Image


def resize_images_to_smallest_height(images):
    # Find the minimum height among all images
    min_height = min(img.height for img in images)

    resized_images = []
    for img in images:
        # Calculate the new width to maintain the aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(min_height * aspect_ratio)

        # Resize the image
        resized_img = img.resize((new_width, min_height))
        resized_images.append(resized_img)

    return resized_images


def concatenate_pil_images_horizontally(images: list[Image.Image]) -> Image.Image:
    resized_images = resize_images_to_smallest_height(images)
    
    # insert white space between
    white_space_width = int(resized_images[0].width * 0.1)
    white_space = Image.new('RGB', (white_space_width, resized_images[0].height), (255, 255, 255))
    images = []
    for image in resized_images:
        images.append(image)
        images.append(white_space)
    images = images[:-1]  # remove the last white space
    
    # Calculate the total width and the maximum height of the new image
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with the appropriate size
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste each image next to each other horizontally
    x_offset = 0
    for img in images:
        # Calculate the y-offset to center the image vertically
        y_offset = (max_height - img.height) // 2
        new_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return new_image
