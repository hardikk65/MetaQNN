

def pool_size(image_size):

    if image_size >= 8:
        return 8
    elif image_size >=4 and image_size < 8:
        return 4
    
    return 1