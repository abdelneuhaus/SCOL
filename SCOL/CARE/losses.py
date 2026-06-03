from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import backend_channels_last

import tensorflow as tf
import numpy as np
from ..utils.tf import keras_import, BACKEND as K


def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n     = K.shape(y_true)[-1]
            mu    = y_pred[...,:n]
            sigma = y_pred[...,n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n     = K.shape(y_true)[1]
            mu    = y_pred[:,:n,...]
            sigma = y_pred[:,n:,...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            y_true = K.cast(y_true, K.floatx())
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss






# def loss_weighted_loss_tf(gamma=2.0, alpha=0.65, epsilon=1e-8):
#     """
#     BASIS
#     Weighted loss combinée avec SSIM pour SMLM sparse.
    
#     Args:
#         gamma: exponent pour augmenter le poids des pixels non-nuls
#         alpha: poids pour équilibrer MSE et SSIM
#         epsilon: pour éviter des poids nuls
#     Returns:
#         Fonction de perte combinée
#     """
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         weight = tf.pow(tf.maximum(y_true, epsilon), gamma)
#         weighted_y_true = y_true * weight
#         weighted_y_pred = y_pred * weight
#         mse_loss = tf.reduce_mean(tf.abs(weighted_y_true - weighted_y_pred))
#         ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=tf.reduce_max(y_true) + epsilon))

#         return alpha * mse_loss + (1 - alpha) * ssim_loss
#     return loss




# def loss_weighted_loss_tf(gamma=2.0, alpha=0.65, epsilon=1e-8):
#     """
#     The one which seems to work FOR FIXED DATA
#     """
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         # Éviter batch vide
#         max_val = tf.reduce_max(y_true)
#         max_val = tf.maximum(max_val, epsilon)
#         # Normalisation sûre
#         y_true_n = y_true / max_val
#         y_pred_n = y_pred / max_val
#         # Clamp pour SSIM
#         y_true_n = tf.clip_by_value(y_true_n, 0.0, 1.0)
#         y_pred_n = tf.clip_by_value(y_pred_n, 0.0, 1.0)

#         # Poids doux
#         weight = tf.pow(y_true_n + epsilon, gamma)
#         weight = tf.clip_by_value(weight, 1.0, 5.0)
#         weight = weight / (tf.reduce_mean(weight) + epsilon)
#         # MSE pondérée
#         mse_loss = tf.reduce_mean(weight * tf.square(y_true_n - y_pred_n))

#         # SSIM protégé
#         ssim = tf.image.ssim(y_true_n, y_pred_n, max_val=1.0, filter_size=7)
#         ssim_loss = 1.0 - tf.reduce_mean(ssim)
#         total_loss = alpha * mse_loss + (1.0 - alpha) * ssim_loss

#         # Sécurité ultime
#         total_loss = tf.where(
#             tf.math.is_finite(total_loss),
#             total_loss,
#             mse_loss
#         )
#         return total_loss
#     return loss









"""
ICI ON TESTE
"""

# def loss_weighted_loss_tf(penalty=50.0):
#     """
#     SPT LOSS
#     """
#     def loss(y_true, y_pred):
#         abs_error = tf.abs(y_true - y_pred)
        
#         is_background = tf.cast(y_true < 1, tf.float32)
        
#         # Le poids vaut 'penalty' sur le fond, et 1.0 sur les vraies PSF
#         weights = 1.0 + (penalty - 1.0) * is_background
        
#         return tf.reduce_mean(abs_error * weights)
#     return loss



# def loss_weighted_loss_tf(signal_weight=10.0):
#     """
#     MAE pondérée
#     """
#     def loss(y_true, y_pred):
#         # Erreur au carré (MSE)
#         error = tf.abs(y_true - y_pred)
#         # On booste uniquement là où y_true est brillant
#         weight = 1.0 + (signal_weight * y_true)
#         return tf.reduce_mean(error * weight)
#     return loss




# def loss_weighted_loss_tf(penalty_fp=5.0, sparsity_weight=0.2, epsilon=1e-8):
#     """
#     Loss optimisée pour une architecture résiduelle n-to-n.
#     Agit indépendamment sur chaque channel temporel.
#     """
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
        
#         # Normalisation locale entre 0.0 et 1.0 pour garantir l'échelle
#         max_val = tf.maximum(tf.reduce_max(y_true), epsilon)
#         y_true_n = tf.clip_by_value(y_true / max_val, 0.0, 1.0)
#         y_pred_n = tf.clip_by_value(y_pred / max_val, 0.0, 1.0)

#         # 1. Carte continue du fond (1.0 = noir absolu, 0.0 = signal max)
#         bg_map = 1.0 - y_true_n
        
#         # 2. Score d'hallucination (Continu)
#         # Ce score culmine uniquement quand (Vrai = Noir) ET (Prédiction = Brillant)
#         # On peut mettre y_pred_n au carré pour ne punir que les grosses hallucinations
#         hallucination_score = bg_map * tf.square(y_pred_n)
        
#         # 3. Poids dynamique (sans aucun seuil)
#         # Le poids vaut 1 sur les vrais signaux, et grimpe jusqu'à (1 + penalty_fp) sur les purs FP
#         weight = 1.0 + (penalty_fp * hallucination_score)
        
#         # 4. Erreur quadratique pondérée
#         mse_weighted = tf.reduce_mean(weight * tf.square(y_true_n - y_pred_n))
        
#         # 5. Sparsité L1 (pour éteindre le bruit de fond global)
#         l1_sparsity = tf.reduce_mean(tf.abs(y_pred_n))

#         return mse_weighted + (sparsity_weight * l1_sparsity)

#     return loss



def loss_weighted_loss_tf(gamma=2.0, alpha=0.65, epsilon=1e-8, base_penalty_fp=2.0):
    """
    The one for FIXED DATA
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Normalisation sûre
        max_val = tf.maximum(tf.reduce_max(y_true), epsilon)
        y_true_n = tf.clip_by_value(y_true / max_val, 0.0, 1.0)
        y_pred_n = tf.clip_by_value(y_pred / max_val, 0.0, 1.0)

        # 2. Estimation robuste du fond (Médiane approximée ou Moyenne des pixels faibles)
        # On utilise les 50% de pixels les plus faibles pour estimer le fond pur
        # sans être pollué par les molécules brillantes
        median_bg_approx = tf.reduce_mean(tf.boolean_mask(y_true_n, y_true_n < tf.reduce_mean(y_true_n)))
        median_bg_approx = tf.maximum(median_bg_approx, epsilon) # Sécurité division par zéro

        # 3. Calcul du ratio dynamique pixel par pixel (Peak / Background)
        # Ce tenseur contient la "vraie" valeur dynamique de l'image
        dynamic_snr_ratio = (y_true_n + epsilon) / median_bg_approx
        
        # Le plafond n'est plus "5.0", c'est la valeur maximale du SNR de l'image
        dynamic_max_cap = tf.reduce_max(dynamic_snr_ratio)

        # 4. Calcul des "Poids doux" (Exponentiels)
        weight = tf.pow(y_true_n + epsilon, gamma)
        
        # 5. Le Clipping Dynamique !
        # Le fond est puni a minima (base_penalty_fp, ex: 1.0 ou 2.0 pour taper plus fort sur les FP)
        # Le plafond s'ajuste au contraste réel de la molécule
        weight = tf.clip_by_value(weight, base_penalty_fp, dynamic_max_cap)
        
        # 6. Auto-Normalisation des poids
        weight = weight / (tf.reduce_mean(weight) + epsilon)
        
        # 7. MSE Pondérée
        mse_loss = tf.reduce_mean(weight * tf.square(y_true_n - y_pred_n))

        # 8. SSIM protégé
        ssim = tf.image.ssim(y_true_n, y_pred_n, max_val=1.0, filter_size=7)
        ssim_loss = 1.0 - tf.reduce_mean(ssim)
        
        total_loss = alpha * mse_loss + (1.0 - alpha) * ssim_loss

        # 9. Sécurité ultime
        total_loss = tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            mse_loss
        )
        return total_loss
        
    return loss