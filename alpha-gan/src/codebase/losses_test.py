"""Tests for GAN losses."""

import tensorflow as tf

import losses

class GANLossesTest(object):

    def init_constants(self):
        self._discriminator_real_outputs_np = [-5.0, 1.4, 12.5, 2.7]
        self._discriminator_gen_outputs_np = [10.0, 4.4, -5.5, 3.6]
        self._weights = 2.3
        self._discriminator_real_outputs = tf.constant(
            self._discriminator_real_outputs_np, dtype=tf.float32)
        self._discriminator_gen_outputs = tf.constant(
            self._discriminator_gen_outputs_np, dtype=tf.float32)

    def testGeneratorAllCorrect(self):
        loss = self._g_loss_fn(self._discriminator_gen_outputs)
        self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
        self.assertEqual(self._generator_loss_name, loss.op.name)
        with self.test_session():
            self.assertAlmostEqual(self._expected_g_loss, loss.eval(), 5)

    def testDiscriminatorAllCorrect(self):
        loss = self._d_loss_fn(
            self._discriminator_real_outputs, self._discriminator_gen_outputs)
        self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
        self.assertEqual(self._discriminator_loss_name, loss.op.name)
        with self.test_session():
            self.assertAlmostEqual(self._expected_d_loss, loss.eval(), 5)

    def testGeneratorLossCollection(self):
        self.assertEqual(0, len(tf.get_collection('collection')))
        self._g_loss_fn(
            self._discriminator_gen_outputs, loss_collection='collection')
        self.assertEqual(1, len(tf.get_collection('collection')))

    def testDiscriminatorLossCollection(self):
        self.assertEqual(0, len(tf.get_collection('collection')))
        self._d_loss_fn(
            self._discriminator_real_outputs, self._discriminator_gen_outputs,
            loss_collection='collection')
        self.assertEqual(1, len(tf.get_collection('collection')))

    def testGeneratorNoReduction(self):
        loss = self._g_loss_fn(
            self._discriminator_gen_outputs, reduction=tf.losses.Reduction.NONE)
        self.assertEqual([4], loss.shape)

    def testDiscriminatorNoReduction(self):
        loss = self._d_loss_fn(
            self._discriminator_real_outputs, self._discriminator_gen_outputs,
            reduction=tf.losses.Reduction.NONE)
        self.assertEqual([4], loss.shape)

    def testGeneratorPatch(self):
        loss = self._g_loss_fn(
            tf.reshape(self._discriminator_gen_outputs, [2, 2]))
        self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
        with self.test_session():
            self.assertAlmostEqual(self._expected_g_loss, loss.eval(), 5)

    def testDiscriminatorPatch(self):
        loss = self._d_loss_fn(
            tf.reshape(self._discriminator_real_outputs, [2, 2]),
            tf.reshape(self._discriminator_gen_outputs, [2, 2]))
        self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
        with self.test_session():
            self.assertAlmostEqual(self._expected_d_loss, loss.eval(), 5)

class MinimaxLossTest(tf.test.TestCase, GANLossesTest):
    """Tests for minimax_xxx_loss."""

    def setUp(self):
        super(MinimaxLossTest, self).setUp()
        self.init_constants()
        self._expected_g_loss = -4.82408
        self._expected_d_loss = 5.83386
        self._generator_loss_name = 'generator_minimax_loss/Neg'
        self._discriminator_loss_name = 'discriminator_minimax_loss/add'
        self._g_loss_fn = losses.minimax_generator_loss
        self._d_loss_fn = losses.minimax_discriminator_loss


if __name__ == '__main__':
    tf.test.main()
