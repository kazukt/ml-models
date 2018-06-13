import functools

import tensorflow as tf

def sigmoid_density_ratio(
    logits,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    if logits is None:
        raise valueError('logits must not be None')
    with tf.name_scope(
        scope, 'sigmoid_density_ratio_loss', [logits, weights]) as scope:
        logits = tf.convert_to_tensor(logits)

        losses = tf.negative(logits)

        return tf.losses.compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)

def minimax_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(scope, 'discriminator_minimax_loss', [
        discriminator_gen_outputs, discriminator_real_outputs,
        real_weights, generated_weights, label_smoothing]) as scope:

        # - log(D(x))
        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs),
            discriminator_real_outputs,
            real_weights,
            label_smoothing,
            scope,
            loss_collection=None,
            reduction=reduction)

        # - log(1 - D(G(z)))
        loss_on_fake = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs),
            discriminator_gen_outputs,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction)

        loss = loss_on_real + loss_on_fake
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
            tf.summary.scalar('discriminator_gen_minimax_loss', loss_on_fake)
            tf.summary.scalar('discriminator_minimax_loss', loss)

    return loss

def minimax_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(scope, 'generator_minimax_loss', [
        discriminator_gen_outputs, weights, label_smoothing]) as scope:
        loss = - minimax_discriminator_loss(
            tf.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs,
            weights,
            weights,
            label_smoothing,
            scope,
            loss_collection,
            reduction,
            add_summaries=False)

        if add_summaries:
            tf.summary.scalar('generator_minimax_loss', loss)

    return loss

def modified_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    return minimax_discriminator_loss(
        discriminator_real_outputs,
        discriminator_gen_outputs,
        real_weights,
        generated_weights,
        label_smoothing,
        scope or 'discriminator_modified_loss',
        loss_collection,
        reduction,
        add_summaries)

def modified_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope,
        'generator_modified_loss',
        [discriminator_gen_outputs, weights, label_smoothing]) as scope:
        loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs,
            weights,
            label_smoothing,
            scope,
            loss_collection,
            reduction)

        if add_summaries:
            tf.summary.scalar('generator_modified_loss', loss)

    return loss

def alphagan_encoder_loss(
    discriminator_gen_outputs,
    data,
    reconstructed_data,
    reconstructed_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope, 'alphagan_encoder_loss',
        [discriminator_gen_outputs, data, reconstructed_data]):
        generation_loss = sigmoid_density_ratio(
            discriminator_gen_outputs,
            loss_collection=None,
            reduction=reduction)

        reconstruction_loss = tf.losses.absolute_difference(
            data,
            reconstructed_data,
            reconstructed_weights,
            loss_collection=None,
            reduction=reduction)

        loss = reconstruction_loss + generation_loss
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar(
                'alphagan_encoder_generation_loss', generation_loss)
            tf.summary.scalar(
                'alphagan_encoder_reconstruction_loss', reconstruction_loss)
            tf.summary.scalar('alphagan_encoder_loss', loss)

        return loss

def alphagan_generator_loss(
    discriminator_gen_outputs,
    discriminator_rec_outputs,
    data,
    reconstructed_data,
    generated_weights=1.0,
    reconstructed_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope, 'alphagan_generator_loss',
        [discriminator_gen_outputs, discriminator_rec_outputs,
         data, reconstructed_data,
         generated_weights, reconstructed_weights]) as scope:
        generation_loss = sigmoid_density_ratio(
            discriminator_gen_outputs,
            generated_weights,
            scope,
            loss_collection=None,
            reduction=reduction)

        reconstruction_density_loss = sigmoid_density_ratio(
            discriminator_rec_outputs,
            reconstructed_weights,
            scope,
            loss_collection=None,
            reduction=reduction)

        reconstruction_loss = tf.losses.absolute_difference(
            data, reconstructed_data,
            reconstructed_weights,
            scope,
            loss_collection=None,
            reduction=reduction)

        loss = reconstruction_loss + reconstruction_density_loss + generation_loss
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar(
                'alphagan_generator_gen_density_ratio_loss', generation_loss)
            tf.summary.scalar(
                'alphagan_generator_rec_density_ratio_loss', reconstruction_density_loss)
            tf.summary.scalar(
                'alphagan_generator_rec_absolute_difference_loss', reconstruction_loss)
            tf.summary.scalar('alphagan_generator_loss', loss)

        return loss

def alphagan_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_rec_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope, 'alphagan_discriminator_loss',
        [discriminator_real_outputs, discriminator_gen_outputs,
         discriminator_rec_outputs, real_weights, generated_weights]) as scope:
        loss_original = modified_discriminator_loss(
            discriminator_real_outputs,
            discriminator_gen_outputs,
            real_weights,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction,
            add_summaries=add_summaries)

        loss_on_rec = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_rec_outputs),
            discriminator_rec_outputs,
            scope=scope,
            loss_collection=None,
            reduction=reduction)

        loss = loss_original + loss_on_rec
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('alphagan_discriminator_rec_loss', loss_on_rec)
            tf.summary.scalar('alphagan_discriminator_loss', loss)

        return loss
