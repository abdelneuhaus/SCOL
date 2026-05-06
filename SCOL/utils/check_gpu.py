import tensorflow as tf

print(f"Version de TensorFlow : {tf.__version__}")

# Vérifie si TensorFlow a été compilé avec CUDA
print(f"Compilé avec CUDA : {tf.test.is_built_with_cuda()}")

# Détails des versions de build (CUDA et cuDNN)
build_info = tf.sysconfig.get_build_info()
print(f"Version de CUDA (au build) : {build_info.get('cuda_version', 'N/A')}")
print(f"Version de cuDNN (au build) : {build_info.get('cudnn_version', 'N/A')}")

# Vérification de la disponibilité du GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) détecté(s) : {len(gpus)}")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("Aucun GPU détecté par TensorFlow.")