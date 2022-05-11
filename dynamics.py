import numpy as np
import os
import tensorflow as tf


DT = 0.0  # Time step [s].
MODEL_URDF = ""  # Path to a URDF model (with D revolute joints).
MAX_TORQUES = []  # Maximum torque for each joint [Nm].
MAX_VELOCITIES = []  # Maximum velocity for each joint [rad/s].
ACCELERATION_FACTORS = []  # (Optional) values to add to the diagonal of the mass matrix [kg*m^2/rad].
TORQUE_LIMIT_RATIO = 0.9  # Fraction of the maximum torques at which the torque limit penalty starts.
VELOCITY_LIMIT_RATIO = 0.9  # Fraction of the maximum velocities at which the velocity limit penalty starts.
REGULARIZER = 0.0  # Regularizer added to the diagonal of the pseudo-Hessian before solving.

_dynamics_op = tf.load_op_library(os.path.join(os.path.dirname(__file__), "dynamics_op.so"))


@tf.custom_gradient
def dynamic_loss_fn(trajectory, q_1, q0):
    """Calculates dynamics-based loss functions with preconditioned gradient over a batch of B trajectories of length T.

    Args:
        trajectory: A trajectory from t=1 to t=N, float64, [B x T x D].
        q_1: The joint configuration at t=-1 (no gradient), float64, [B x D].
        q0: The joint configuration at t=0 (no gradient), float64, [B x D].

    Returns:
        The trajectory that should be used in custom loss calculations, such that the gradient of the
        final loss w.r.t. the trajectory gets backpropagated through this function and therefore preconditioned.

        A tuple of three loss values:
        1) 0.5 * sum((torque_j(t) / max_torques[j])^2)
        2) 0.5 * sum(max((abs(torque_j(t)) / max_torques[j] - torque_limit_ratio) / (1 - torque_limit_ratio), 0)^2)
        3) 0.5 * sum(max((abs(q_j(t + 1) - q_j(t)) / (dt * max_velocities[j]) - velocity_limit_ratio) / (1 - velocity_limit_ratio), 0)^2)
        These values may only be used in linear combination in the final loss.

        The return value backward is invisible to the caller because it is consumed by the custom_gradient decorator.
    """
    dt = tf.constant(DT, dtype=tf.float64)
    max_torques = tf.constant(MAX_TORQUES, dtype=tf.float64)
    max_velocities = tf.constant(MAX_VELOCITIES, dtype=tf.float64)
    acceleration_factors = tf.constant(ACCELERATION_FACTORS, dtype=tf.float64)
    torque_limit_ratio = tf.constant(TORQUE_LIMIT_RATIO, dtype=tf.float64)
    velocity_limit_ratio = tf.constant(VELOCITY_LIMIT_RATIO, dtype=tf.float64)
    regularizer = tf.constant(REGULARIZER, dtype=tf.float64)

    def backward(*args):
        return _dynamics_op.dynamic_loss_gradient(trajectory, q_1, q0, dt,
                                                  max_torques,
                                                  max_velocities,
                                                  acceleration_factors,
                                                  torque_limit_ratio,
                                                  velocity_limit_ratio,
                                                  regularizer,
                                                  *args,
                                                  model=MODEL_URDF), tf.zeros_like(q_1), tf.zeros_like(q0)

    dynamic_losses = _dynamics_op.dynamic_loss(trajectory, q_1, q0, dt,
                                               max_torques,
                                               max_velocities,
                                               acceleration_factors,
                                               torque_limit_ratio,
                                               velocity_limit_ratio,
                                               model=MODEL_URDF)

    return (trajectory, dynamic_losses), backward
