/**
 * @file op.cc
 *
 * This file registers the DynamicLoss and DynamicLossGradient ops.
 *
 * @author Arne Hasselbring
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("DynamicLoss")
  .Attr("model: string")
  .Input("trajectory: double")
  .Input("q_1: double")
  .Input("q0: double")
  .Input("dt: double")
  .Input("max_torques: double")
  .Input("max_velocities: double")
  .Input("acceleration_factors: double")
  .Input("torque_limit_ratio: double")
  .Input("velocity_limit_ratio: double")
  .Output("torque_cost: double")
  .Output("torque_limit: double")
  .Output("velocity_limit: double")
  .SetShapeFn([](shape_inference::InferenceContext* c)
  {
    c->set_output(0, c->Scalar());
    c->set_output(1, c->Scalar());
    c->set_output(2, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("DynamicLossGradient")
  .Attr("model: string")
  .Input("trajectory: double")
  .Input("q_1: double")
  .Input("q0: double")
  .Input("dt: double")
  .Input("max_torques: double")
  .Input("max_velocities: double")
  .Input("acceleration_factors: double")
  .Input("torque_limit_ratio: double")
  .Input("velocity_limit_ratio: double")
  .Input("regularizer: double")
  .Input("gradient_trajectory: double")
  .Input("gradient_torque_cost: double")
  .Input("gradient_torque_limit: double")
  .Input("gradient_velocity_limit: double")
  .Output("gradient: double")
  .SetShapeFn([](shape_inference::InferenceContext* c)
  {
    shape_inference::ShapeHandle trajectory_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &trajectory_shape));
    c->set_output(0, trajectory_shape);
    return Status::OK();
  });
