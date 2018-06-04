import functools

import tensorflow as tf

def minimax_discriminator_loss(
    discriminator_real_outputs,
    discriminator_fake_outputs,
    real_weights=1.0,
    generated_weights=1.0
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(scope, 'discriminator_minimax_loss', [
        discriminator_fake_outputs, discriminator_real_outputs,
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
            tf.zeros_like(discriminator_fake_output),
            discriminator_fake_outputs,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction)

        loss = loss_on_real + loss_on_fake
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
            tf.summary.scalar('discriminator_fake_minimax_loss', loss_on_fake)
            tf.summary.scalar('discriminator_minimax_loss', loss)

    return loss

def minimax_generator_loss(
    discriminator_fake_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(scope, 'discriminator_generator_loss', [
        discriminator_fake_outputs, weights, label_smoothing]) as scope:
        loss = - minimax_discriminator_loss(
            tf.ones_like(discriminator_fake_outputs),
            discriminator_fake_outputs,
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
    discriminator_fake_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    return minimax_discriminator_loss(
        discriminator_real_outputs,
        discriminator_fake_outputs,
        real_weights,
        generated_weights,
        label_smoothing,
        scope or 'discriminator_modified_loss',
        loss_collection,
        reduction,
        add_summaries)

def modified_generator_loss(
    discriminator_fake_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope,
        'generator_modified_loss',
        [discriminator_fake_outputs, weights, label_smoothing]) as scope:
        loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_fake_outputs),
            discriminator_fake_outputs,
            weights,
            label_smoothing,
            scope,
            loss_collection,
            reduction)

        if add_summaries:
            tf.summary.scalar('generator_modified_loss', loss)

    return loss

def density_ratio_loss(
    discriminator_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Original density ratio loss.
    """
    with tf.name_scope(
        scope, 'density_ratio_loss',
        [discriminator_outputs, real_weights,
        generated_weights, label_smoothing]) as scope:

        loss_class0 = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_outputs),
            discriminator_outputs,
            weights,
            label_smoothing,
            scope,
            loss_collection=None,
            reduction=reduction)

        loss_class1 = - tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_outputs),
            discriminator_outputs,
            weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction)

        loss = loss_class0 + loss_class1
        tf.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.summary.scalar('density_ratio_loss', loss)


    return loss

def density_ratio_generator_loss(
    discriminator_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope,
        'generator_density_ratio_loss',
        [discriminator_outputs, weights, label_smoothing]) as scope:
        loss = density_ratio_loss(
            discriminator_outputs,
            weights,
            label_smoothing,
            scope,
            loss_collection,
            reduction,
            add_summaries=False)

        if add_summaries:
            tf.summary.scalar('generator_density_ratio_loss', loss)

    return loss

def density_ratio_encoder_loss(
    discriminator_outputs,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope,
        'encoder_density_ratio_loss',
        [discriminator_outputs, weights, label_smoothing]) as scope:
        loss = density_ratio_loss(
            discriminator_outputs=,
            weights,
            label_smoothing,
            scope,
            loss_collection,
            reduction,
            add_summaries=False)

        if add_summaries:
            tf.summary.scalar('encoder_density_ratio_loss', loss)

    return loss

def least_absolute_loss(
    data,
    reconstructed_data,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    with tf.name_scope(
        scope,
        'least_absolute_loss',
        [data, reconstructed_data, weights]) as scope:
        loss = tf.losses.absolute_difference(
            data,
            reconstructed_data,
            weights,
            scope,
            loss_collection,
            reduction)

        if add_summaries:
            tf.summary.scalar('least_absolute_loss', loss)

    return loss

def alphagan_encoder_loss(
    discriminator_gen_outputs,
    data,
    reconstructed_data,
    encoder_loss_fn=density_ratio_encoder_loss,
    reconstruction_loss_fn=least_absolute_loss,
    reconstruction_loss_weight=1.0,
    add_summaries=False):
    loss_on_enc = encoder_loss_fn(discriminator_gen_outputs)
    rec_loss = reconstruction_loss_fn(
        data, reconstructed_data, reconstruction_loss_weight)

    loss = loss_on_enc + rec_loss

    if add_summaries:
        tf.summary.scalar('alphagan_encoder_loss', loss)

    return loss

def alphagan_generator_loss(
    discriminator_gen_outputs,
    discriminator_rec_outputs,
    data,
    reconstructed_data,
    generator_loss_fn=density_ratio_generator_loss,
    reconstruction_loss_fn=least_absolute_loss,
    reconstruction_loss_weight=1.0,
    add_summaries=False):
    loss_on_gen = generator_loss_fn(discriminator_gen_outputs)

    loss_on_rec = generator_loss_fn(discriminator_rec_outputs)

    rec_loss = reconstruction_loss_fn(
        data, reconstructed_data, reconstruction_loss_weight)

    loss = rec_loss + loss_on_rec + loss_on_gen

    if add_summaries:
        tf.summary.scalar('alphagan_generator_loss', loss)

    return loss

def alphagan_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_rec_outputs,
    discriminator_fn=modified_discriminator_loss,
    add_summaries=False):
    dis_loss = discriminator_fn(
        discriminator_real_outputs, discriminator_gen_outputs)

    loss_on_rec = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_rec_outputs),
        discriminator_rec_outputs)

    loss = dis_loss + loss_on_rec

    if add_summaries:
        tf.summary.scalar('alphagan_discriminator_loss', loss)

    return loss

def alphagan_code_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    add_summaries=False):
    loss = modified_discriminator_loss(
        discriminator_real_outputs,
        discriminator_gen_outputs,
        scope='alphagan_code_discriminator_loss')

    if add_summaries:
        tf.summary.scalar('alphagan_code_discriminator_loss', loss)

    return loss
