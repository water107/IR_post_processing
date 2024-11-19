from PIL import Image
from numpy import average, dot, linalg

def get_thum(image, size=(180, 250), greyscale=False):
    image = image.resize(size, Image.LANCZOS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

def cos_similarity(pic1, pic2):
    try:
        image1 = Image.open(pic1)
        image2 = Image.open(pic2)
    except (OSError, IOError) as e:
        print(f"无法打开图像文件: {e}")

    cosine = image_similarity_vectors_via_numpy(image1, image2)
    print('图片余弦相似度：', cosine)
    return cosine

if __name__ == '__main__':
    pic1 = 'gray_fly.png'
    pic2 = 'gray_apparent.png'
    cos_similarity(pic1, pic2)

    