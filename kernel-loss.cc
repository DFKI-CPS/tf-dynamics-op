/**
 * @file kernel-loss.cc
 *
 * This file defines the kernel for the DynamicLoss op.
 *
 * @author Arne Hasselbring
 */

#include "tensorflow/core/framework/op_kernel.h"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <mutex>

using namespace tensorflow;

class DynamicLossOp : public OpKernel
{
public:
  explicit DynamicLossOp(OpKernelConstruction* context) :
    OpKernel(context)
  {
    std::string model_path;
    OP_REQUIRES_OK(context, context->GetAttr("model", &model_path));

    pinocchio::urdf::buildModel(model_path, model);
    data = pinocchio::Data(model);

    OP_REQUIRES(context, model.nq == model.nv, errors::InvalidArgument("Invalid robot model."));
  }

  void Compute(OpKernelContext* context) override
  {
    OP_REQUIRES(context, context->num_inputs() == 9, errors::InvalidArgument("Wrong number of inputs."));
    const Tensor& trajectory_tensor = context->input(0);
    const Tensor& q_1_tensor = context->input(1);
    const Tensor& q0_tensor = context->input(2);
    const Tensor& dt_tensor = context->input(3);
    const Tensor& max_torques_tensor = context->input(4);
    const Tensor& max_velocities_tensor = context->input(5);
    const Tensor& acceleration_factors_tensor = context->input(6);
    const Tensor& torque_limit_ratio_tensor = context->input(7);
    const Tensor& velocity_limit_ratio_tensor = context->input(8);

    const auto D = model.nq;

    OP_REQUIRES(context,
                trajectory_tensor.shape().dims() == 3 &&
                trajectory_tensor.shape().dim_size(2) == D &&
                q_1_tensor.shape().dims() == 2 &&
                q_1_tensor.shape().dim_size(1) == D &&
                q0_tensor.shape().dims() == 2 &&
                q0_tensor.shape().dim_size(1) == D &&
                trajectory_tensor.shape().dim_size(0) == q_1_tensor.shape().dim_size(0) &&
                trajectory_tensor.shape().dim_size(0) == q0_tensor.shape().dim_size(0) &&
                dt_tensor.shape().dims() == 0 &&
                max_torques_tensor.shape().dims() == 1 &&
                max_torques_tensor.shape().dim_size(0) == D &&
                max_velocities_tensor.shape().dims() == 1 &&
                max_velocities_tensor.shape().dim_size(0) == D &&
                (acceleration_factors_tensor.shape().dims() == 0 ||
                 (acceleration_factors_tensor.shape().dims() == 1 && acceleration_factors_tensor.shape().dim_size(0) == D)) &&
                torque_limit_ratio_tensor.shape().dims() == 0 &&
                velocity_limit_ratio_tensor.shape().dims() == 0,
                errors::InvalidArgument("Wrong shape(s)."));

    OP_REQUIRES(context, trajectory_tensor.shape().dim_size(1) >= 2, errors::InvalidArgument("Trajectory too short."));

    // Allocate the loss outputs which are scalars.
    Tensor* torque_cost_output_tensor = nullptr, * torque_limit_output_tensor = nullptr, * velocity_limit_output_tensor = nullptr;
    TensorShape loss_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(static_cast<int64_t*>(nullptr), 0, &loss_shape));
    OP_REQUIRES_OK(context, context->allocate_output(0, loss_shape, &torque_cost_output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, loss_shape, &torque_limit_output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, loss_shape, &velocity_limit_output_tensor));

    // Get references to the Eigen objects.
    auto trajectory = trajectory_tensor.tensor<double, 3>();
    auto q_1 = q_1_tensor.matrix<double>();
    auto q0 = q0_tensor.matrix<double>();
    const double dt = *dt_tensor.scalar<double>().data();
    Eigen::Map<const Eigen::ArrayXd> max_torques(max_torques_tensor.vec<double>().data(), D);
    Eigen::Map<const Eigen::ArrayXd> max_velocities(max_velocities_tensor.vec<double>().data(), D);
    const double* acceleration_factors = acceleration_factors_tensor.shape().dims() > 0 ? acceleration_factors_tensor.vec<double>().data() : nullptr;
    const double torque_limit_ratio = *torque_limit_ratio_tensor.scalar<double>().data();
    const double velocity_limit_ratio = *velocity_limit_ratio_tensor.scalar<double>().data();
    auto torque_cost_output = torque_cost_output_tensor->scalar<double>();
    auto torque_limit_output = torque_limit_output_tensor->scalar<double>();
    auto velocity_limit_output = velocity_limit_output_tensor->scalar<double>();

    const double dt_squared = dt * dt;

    // B = batch size, T = length of the trajectory
    const auto B = trajectory_tensor.shape().dim_size(0);
    const auto T = trajectory_tensor.shape().dim_size(1);

    // pinocchio writes the result of the RNEA here.
    auto& tau = data.tau;

    // The loss terms are calculated individually.
    double torque_cost = 0.0;
    double torque_limit = 0.0;
    double velocity_limit = 0.0;

    for(int64_t i = 0; i < B; ++i)
    {
      const auto* trajectory_ptr = trajectory.data() + i * T * D;
      for(int64_t j = 0; j < T; ++j)
      {
        // The first torque is calculated around q0 (actually $q_0$), so the previous element is q_1 (actually $q_{-1}$) and the next element is the first step of the trajectory (actually $q_1$, but has index 0 in memory).
        Eigen::Map<const Eigen::VectorXd> q(j ? (trajectory_ptr + (j - 1) * D) : q0.data() + i * D, D);
        Eigen::Map<const Eigen::VectorXd> q_next(trajectory_ptr + j * D, D);
        Eigen::Map<const Eigen::VectorXd> q_prev(j >= 2 ? (trajectory_ptr + (j - 2) * D) : (j == 1 ? q0.data() + i * D : q_1.data() + i * D), D);
        const Eigen::VectorXd v = (q_next - q) / dt;
        const Eigen::VectorXd a = (q_next + q_prev - 2.0 * q) / dt_squared;

        std::lock_guard<std::mutex> lg(mutex);
        pinocchio::rnea(model, data, q, v, a);

        if(acceleration_factors)
          tau.array() += a.array() * Eigen::Map<const Eigen::ArrayXd>(acceleration_factors, D);

        tau.array() /= max_torques;

        torque_cost += tau.squaredNorm();
        torque_limit += (tau.array().abs() - torque_limit_ratio).cwiseMax(0.0).matrix().squaredNorm();
        velocity_limit += (v.array().abs() / max_velocities - velocity_limit_ratio).cwiseMax(0.0).matrix().squaredNorm();
      }
      {
        // The last torque has the boundary condition that $q_{T} = q_{T+1}$, so v is 0 and a is just the negative velocity divided by dt.
        Eigen::Map<const Eigen::VectorXd> q(trajectory_ptr + (T - 1) * D, D);
        const Eigen::VectorXd a = (Eigen::Map<const Eigen::VectorXd>(trajectory_ptr + (T - 2) * D, D) - q) / dt_squared;

        std::lock_guard<std::mutex> lg(mutex);
        pinocchio::rnea(model, data, q, Eigen::VectorXd::Zero(D), a);

        if(acceleration_factors)
          tau.array() += a.array() * Eigen::Map<const Eigen::ArrayXd>(acceleration_factors, D);

        tau.array() /= max_torques;

        torque_cost += tau.squaredNorm();
        torque_limit += (tau.array().abs() - torque_limit_ratio).cwiseMax(0.0).matrix().squaredNorm();
        // Since the velocity is 0 here, the velocity limit component is 0, too.
      }
    }

    // Incorporate normalization factors such that being at the limit equals 1.
    torque_limit /= (1.0 - torque_limit_ratio) * (1.0 - torque_limit_ratio);
    velocity_limit /= (1.0 - velocity_limit_ratio) * (1.0 - velocity_limit_ratio);

    // Average over batch. Factor 1/2 is due to quadratic form (if it wasn't here there would be a 2 in the gradient).
    *torque_cost_output.data() = 0.5 * torque_cost / B;
    *torque_limit_output.data() = 0.5 * torque_limit / B;
    *velocity_limit_output.data() = 0.5 * velocity_limit / B;
  }

private:
  pinocchio::Model model; ///< The robot model.
  pinocchio::Data data; ///< The data on which pinocchio operates.
  std::mutex mutex; ///< Mutex to synchronize access to \c data (the TF docs say that \c Compute can be called from multiple threads simultaneously).
};

REGISTER_KERNEL_BUILDER(Name("DynamicLoss").Device(DEVICE_CPU), DynamicLossOp);
