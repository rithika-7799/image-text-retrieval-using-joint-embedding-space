# get the hard positive, semi_hard_negative and hard negative
import tensorflow as tf
params = []
with open("hyperparams.txt") as f:
    params = f.read().split(",")
params = [float(p) for p in params]
margin,margin1,margin2,margin3, lambda1, lambda2, lambda3 =params #0.1,0.15,0.1,0.2,2,1,0.5
print("Params: ", margin,margin1,margin2,margin3, lambda1, lambda2, lambda3)
def _pairwise_distances_cos_(embeddings):

    distances = tf.reduce_sum(tf.multiply(embeddings, embeddings))

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)
    mask = tf.compat.v1.to_float(tf.equal(distances, 0.0))
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    distances = distances * (1.0 - mask)

    return distances


def _pairwise_distances(embeddings, embeddings2, squared=False):
    embeddings = tf.cast(embeddings, tf.float32)#embeddings.astype('float32')
    embeddings2 = tf.cast(embeddings2, tf.float32)#embeddings2.astype('float32')
    # print(embeddings.shape)
    # print(embeddings2.shape)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings2))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.compat.v1.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Check that i and j are distinct
    #print()
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0),
                            tf.expand_dims(labels, 1))

    # Combine the two masks
    labels_equal=tf.reshape(labels_equal, [32, 32])
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):

    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0),
                            tf.expand_dims(labels, 1))
    labels_equal=tf.reshape(labels_equal, [32, 32])

    mask = tf.logical_not(labels_equal)

    return mask


def semi_hard_(margin, pairwise_dist, anchor_negative_dist, hardest_positive_dist, hardest_negative_dist):
    X = pairwise_dist
    lo = tf.get_static_value(hardest_positive_dist)
    if lo == 0:
        hi = tf.get_static_value(tf.reduce_min(anchor_negative_dist, axis=0))

    else:
        hi = tf.get_static_value(
            tf.get_static_value(hardest_positive_dist)+margin)

    mask = tf.math.logical_and(tf.math.greater(
        X, lo), tf.math.less_equal(X, hi))
    X1 = tf.boolean_mask(X, mask)
    is_empty = tf.equal(tf.size(X1), 0)
    if is_empty == True:
        # print("here")
        temp = tf.get_static_value(hardest_negative_dist)
        return tf.cast(temp[0], tf.float32)#temp[0].astype('float32')
    else:
        return tf.get_static_value(tf.reduce_max(X1))


def image_image_anchor_dist(pairwise_dist, mask_anchor_positive, mask_anchor_negative, margin):
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    hardest_positive_dist = tf.reduce_max(
        anchor_positive_dist, axis=1, keepdims=True)
    # Negative distance calculation
    max_anchor_negative_dist = tf.reduce_max(
        pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + \
        max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = tf.reduce_min(
        anchor_negative_dist, axis=1, keepdims=True)
    return hardest_positive_dist, hardest_negative_dist #, semi_hard_negative_


def text_text_anchor_dist(pairwise_dist_text, mask_anchor_positive, mask_anchor_negative, margin):
    anchor_positive_dist_text = tf.multiply(
        mask_anchor_positive, pairwise_dist_text)
    hardest_positive_dist_text = tf.reduce_max(
        anchor_positive_dist_text, axis=1, keepdims=True)  # closest image

    max_anchor_negative_dist_text = tf.reduce_max(
        pairwise_dist_text, axis=1, keepdims=True)
    anchor_negative_dist_text = pairwise_dist_text + \
        max_anchor_negative_dist_text * (1.0 - mask_anchor_negative)
    hardest_negative_dist_text = tf.reduce_min(
        anchor_negative_dist_text, axis=1, keepdims=True)
    '''
    semi_hard_negative_text = []
    for i in range(hardest_positive_dist_text.shape[0]):
        t2 = semi_hard_(margin, pairwise_dist_text[i], anchor_negative_dist_text[i],
                        hardest_positive_dist_text[i], hardest_negative_dist_text[i])
        semi_hard_negative_text.append(t2)
    semi_hard_negative_text = tf.convert_to_tensor(
        semi_hard_negative_text, dtype=tf.float32)
    semi_hard_negative_text = tf.expand_dims(semi_hard_negative_text, 1)
    '''

    return hardest_positive_dist_text, hardest_negative_dist_text  #, semi_hard_negative_text


def image_text_anchor_dist(pairwise_dist_i_text, mask_anchor_positive, mask_anchor_negative, margin):
    anchor_positive_dist_i_text = tf.multiply(
        mask_anchor_positive, pairwise_dist_i_text)
    hardest_positive_dist_i_text = tf.reduce_max(
        anchor_positive_dist_i_text, axis=1, keepdims=True)

    max_anchor_negative_dist_i_text = tf.reduce_max(
        pairwise_dist_i_text, axis=1, keepdims=True)
    anchor_negative_dist_i_text = pairwise_dist_i_text + \
        max_anchor_negative_dist_i_text * (1.0 - mask_anchor_negative)
    hardest_negative_dist_i_text = tf.reduce_min(
        anchor_negative_dist_i_text, axis=1, keepdims=True)
    '''
    semi_hard_negative_i_text = []
    for i in range(hardest_positive_dist_i_text.shape[1]):
        t2 = semi_hard_(margin, pairwise_dist_i_text[i], anchor_negative_dist_i_text[i],
                        hardest_positive_dist_i_text[i], hardest_negative_dist_i_text[i])
        semi_hard_negative_i_text.append(t2)
    semi_hard_negative_i_text = tf.convert_to_tensor(
        semi_hard_negative_i_text, dtype=tf.float32)
    semi_hard_negative_i_text = tf.expand_dims(semi_hard_negative_i_text, 1)
    '''
    return hardest_positive_dist_i_text, hardest_negative_dist_i_text   #, semi_hard_negative_i_text


def text_image_anchor_dist(pairwise_dist_t_image, mask_anchor_positive, mask_anchor_negative, margin):

    # pairwise_dist_t_image = _pairwise_distances(text_embed,image_embed,squared=False)
    anchor_positive_dist_t_image = tf.multiply(
        mask_anchor_positive, pairwise_dist_t_image)
    hardest_positive_dist_t_image = tf.reduce_max(
        anchor_positive_dist_t_image, axis=1, keepdims=True)  # closest image
    max_anchor_negative_dist_t_image = tf.reduce_max(
        pairwise_dist_t_image, axis=1, keepdims=True)
    anchor_negative_dist_t_image = pairwise_dist_t_image + \
        max_anchor_negative_dist_t_image * (1.0 - mask_anchor_negative)
    hardest_negative_dist_t_image = tf.reduce_min(
        anchor_negative_dist_t_image, axis=1, keepdims=True)
    #tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    '''
    semi_hard_negative_t_image = []
    for i in range(hardest_positive_dist_t_image.shape[0]):
        t2 = semi_hard_(margin, pairwise_dist_t_image[i], anchor_negative_dist_t_image[i],
                        hardest_positive_dist_t_image[i], hardest_negative_dist_t_image[i])
        semi_hard_negative_t_image.append(t2)
    semi_hard_negative_t_image = tf.convert_to_tensor(
        semi_hard_negative_t_image, dtype=tf.float32)
    semi_hard_negative_t_image = tf.expand_dims(semi_hard_negative_t_image, 1)
    '''

    return hardest_positive_dist_t_image, hardest_negative_dist_t_image   #, semi_hard_negative_t_image


# (embed,margin,lambda1,lambda2,lambda3): ### To be replaced with the combined embedding table batch
def loss_function_positive_semi_hard_neg(labels, embed):
    
    # print(embed.shape)
    # print(labels.shape)
    
    image_embed = tf.cast(embed[:, 0:64], tf.float32)#embed[:, 0:128]#.astype('float32')
    text_embed = tf.cast(embed[:, 64:], tf.float32)#embed[:, 128:]#.astype('float32')
    # print(image_embed.shape)
    # print(text_embed.shape)
    
    # Creating tensors  to track matching and mismatching labels
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.compat.v1.to_float(mask_anchor_positive)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.compat.v1.to_float(mask_anchor_negative)


    pairwise_dist = _pairwise_distances(image_embed, image_embed, squared=False)  # extracts image
    hardest_positive_dist, hardest_negative_dist = image_image_anchor_dist(
        pairwise_dist, mask_anchor_positive, mask_anchor_negative, margin)
    
    
    # Text-text distance
    pairwise_dist_text = _pairwise_distances(
        text_embed, text_embed, squared=False)
    hardest_positive_dist_text, hardest_negative_dist_text = text_text_anchor_dist(
        pairwise_dist_text, mask_anchor_positive, mask_anchor_negative, margin)

    # Image-text anchor
    pairwise_dist_i_text = _pairwise_distances(
        image_embed, text_embed, squared=False)
    hardest_positive_dist_i_text, hardest_negative_dist_i_text = image_text_anchor_dist(
        pairwise_dist_i_text, mask_anchor_positive, mask_anchor_negative, margin)
   # print(hardest_positive_dist_i_text)
   # print(pairwise_dist_i_text)

    # Text -Image anchor
    pairwise_dist_t_image = _pairwise_distances(
        text_embed, image_embed, squared=False)
    hardest_positive_dist_t_image, hardest_negative_dist_t_image = text_image_anchor_dist(
        pairwise_dist_t_image, mask_anchor_positive, mask_anchor_negative, margin)
    # print(hardest_positive_dist_t_image)
    # print(semi_hard_negative_t_image)

    # calculate triplet loss
    triplet_loss =  lambda2*tf.maximum(hardest_positive_dist - hardest_negative_dist + margin2, 0.0)+ \
    tf.maximum(hardest_positive_dist_i_text - hardest_negative_dist_i_text + margin, 0.0) +\
      lambda3 * tf.maximum(hardest_positive_dist_text - hardest_negative_dist_text + margin3, 0.0) + \
      lambda1*tf.maximum(hardest_positive_dist_t_image - \
                           hardest_negative_dist_t_image + margin1, 0.0)    
    
    '''
    tf.maximum(hardest_positive_dist_i_text - hardest_negative_dist_i_text + margin, 0.0) +       lambda2*tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0) + \
        lambda3 * tf.maximum(hardest_positive_dist_text - hardest_negative_dist_text + margin, 0.0) + \
        lambda1*tf.maximum(hardest_positive_dist_t_image -
                           hardest_negative_dist_t_image + margin, 0.0)
                           '''
    #triplet_loss = tf.reduce_mean(triplet_loss)
    # hardest_positive_dist,hardest_negative_dist,semi_hard_negative_,
    # lambda1*tf.maximum(hardest_positive_dist - semi_hard_negative_ + margin, 0.0)+ tf.maximum(hardest_positive_dist_i_text - semi_hard_negative_i_text + margin, 0.0)  # tf.maximum(hardest_positive_dist_i_text - semi_hard_negative_i_text + margin, 0.0)

    
    
    # print(triplet_loss.shape)
    return triplet_loss
