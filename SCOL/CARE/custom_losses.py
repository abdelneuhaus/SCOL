import tensorflow as tf


def loss_custom_fixed_data(gamma=2.0, alpha=0.65, epsilon=1e-8):
    """
    The one which seems to work FOR FIXED DATA
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Éviter batch vide
        max_val = tf.reduce_max(y_true)
        max_val = tf.maximum(max_val, epsilon)
        # Normalisation sûre
        y_true_n = y_true / max_val
        y_pred_n = y_pred / max_val
        # Clamp pour SSIM
        y_true_n = tf.clip_by_value(y_true_n, 0.0, 1.0)
        y_pred_n = tf.clip_by_value(y_pred_n, 0.0, 1.0)

        # Poids doux
        weight = tf.pow(y_true_n + epsilon, gamma)
        weight = tf.clip_by_value(weight, 1.0, 5.0)
        weight = weight / (tf.reduce_mean(weight) + epsilon)
        # MSE pondérée
        mse_loss = tf.reduce_mean(weight * tf.square(y_true_n - y_pred_n))

        # SSIM protégé
        ssim = tf.image.ssim(y_true_n, y_pred_n, max_val=1.0, filter_size=7)
        ssim_loss = 1.0 - tf.reduce_mean(ssim)
        total_loss = alpha * mse_loss + (1.0 - alpha) * ssim_loss

        # Sécurité ultime
        total_loss = tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            mse_loss
        )
        return total_loss
    return loss



def loss_custom_spt_data(penalty=50.0):
    """
    SPT LOSS
    """
    def loss(y_true, y_pred):
        abs_error = tf.abs(y_true - y_pred)
        
        is_background = tf.cast(y_true < 1, tf.float32)
        
        # Le poids vaut 'penalty' sur le fond, et 1.0 sur les vraies PSF
        weights = 1.0 + (penalty - 1.0) * is_background
        
        return tf.reduce_mean(abs_error * weights)
    return loss



def loss_custom_fixed_data_v2(gamma=2.0, alpha=0.65, epsilon=1e-8, base_penalty_fp=2.0):
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