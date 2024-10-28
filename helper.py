import pandas as pd

_data = None



def load_data():
    global _data

    if _data is None:
        _data = pd.read_csv("q_val.csv")
    
    return _data


def pool_size(image_size):

    if image_size >= 8:
        return 8
    elif image_size >=4 and image_size < 8:
        return 4
    
    return 1


def calculate_image_size(image_size,kernel_size,strides)->int:
    image_size = 1 + (image_size - kernel_size)/strides
    return int(image_size)




if __name__ == "__main__":
    print(calculate_image_size(32,3,1))






