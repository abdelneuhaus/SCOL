import json
import numpy as np
import matplotlib.pyplot as plt


def save_data(points, filename):
    filename = filename.replace('.tif','')
    dictionary = dict()
    for i in range(len(points)):
        dictionary[i] = {
            'coordinates': points[i]['coordinates'],
            'intensity': int(points[i]['intensity']),
            'on_times': np.array(points[i]['on_times'], dtype='uint16').tolist(),
            'shift': np.array(points[i]['shift'], dtype='uint16').tolist()
        }

    json_object = json.dumps(dictionary, indent = 4)
  
    with open(filename+".json", "w") as outfile:
        outfile.write(json_object)

    # with open(str(filename)+'.txt', 'w') as f:
    #     f.write('id \t')
    #     f.write('approximative coordinates (x,y) \t')
    #     f.write('blinking frames \n')
    #     for line in points.keys():
    #         f.write(str(line))
    #         f.write('\t')
    #         f.write(str(tuple(ti for ti in points[line]['coordinates'])[::-1]))
    #         f.write('\t')
    #         f.write(str([x+1 for x in points[line]['on_times']]))
    #         f.write('\n')





def plot_points_image(points, image_shape=(512, 512), point_radius=1):
    """
    Affiche tous les points sur une seule image.

    :param points: Dictionnaire ou liste de points contenant les 'coordinates'.
    :param image_shape: Tuple (hauteur, largeur) de l'image de sortie.
    :param point_radius: Rayon du point dessiné (en pixels).
    """
    image = np.zeros(image_shape, dtype=np.uint8)

    for p in points.values():
        y, x = map(int, p['coordinates'])  # on suppose que coords = (y, x)
        y, x = y*5, x*5
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            # Dessine un point (optionnellement plus grand avec un petit carré)
            image[max(0, y - point_radius):min(image_shape[0], y + point_radius + 1),
                  max(0, x - point_radius):min(image_shape[1], x + point_radius + 1)] = 255

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title("Image des points / mask")
    plt.axis('off')
    plt.show()