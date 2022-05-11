/**
 * @file kernel-loss-gradient.cc
 *
 * This file defines the kernel for the DynamicLossGradient op.
 *
 * @author Arne Hasselbring
 */

#include "tensorflow/core/framework/op_kernel.h"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <algorithm>
#include <cmath>
#include <mutex>

using namespace tensorflow;

// Imported from LAPACK, solves a linear equation system with a real symmetric positive definite band matrix:
// see https://www.netlib.org/lapack/explore-html/de/d49/group__double_o_t_h_e_rsolve_ga9c26c8344bc125d78d6a33a22459169c.html
extern "C" void dpbsv_(const char* uplo, const int* n, const int* kd, const int* nrhs, double* ab, const int* ldab, double* b, const int* ldb, int* info);

class DynamicLossGradientOp : public OpKernel
{
public:
  explicit DynamicLossGradientOp(OpKernelConstruction* context) :
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
    OP_REQUIRES(context, context->num_inputs() == 14, errors::InvalidArgument("Wrong number of inputs."));
    const Tensor& trajectory_tensor = context->input(0);
    const Tensor& q_1_tensor = context->input(1);
    const Tensor& q0_tensor = context->input(2);
    const Tensor& dt_tensor = context->input(3);
    const Tensor& max_torques_tensor = context->input(4);
    const Tensor& max_velocities_tensor = context->input(5);
    const Tensor& acceleration_factors_tensor = context->input(6);
    const Tensor& torque_limit_ratio_tensor = context->input(7);
    const Tensor& velocity_limit_ratio_tensor = context->input(8);
    const Tensor& regularizer_tensor = context->input(9);
    const Tensor& gradient_trajectory_tensor = context->input(10);
    const Tensor& gradient_torque_cost_tensor = context->input(11);
    const Tensor& gradient_torque_limit_tensor = context->input(12);
    const Tensor& gradient_velocity_limit_tensor = context->input(13);

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
                velocity_limit_ratio_tensor.shape().dims() == 0 &&
                regularizer_tensor.shape().dims() == 0 &&
                (gradient_trajectory_tensor.shape().dims() == 0 || gradient_trajectory_tensor.shape() == trajectory_tensor.shape()) &&
                gradient_torque_cost_tensor.shape().dims() == 0 &&
                gradient_torque_limit_tensor.shape().dims() == 0 &&
                gradient_velocity_limit_tensor.shape().dims() == 0,
                errors::InvalidArgument("Wrong shape(s)."));

    OP_REQUIRES(context, trajectory_tensor.shape().dim_size(1) >= 2, errors::InvalidArgument("Trajectory too short."));

    // Allocate the gradient output which has the same shape as the trajectory.
    Tensor* gradient_output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, trajectory_tensor.shape(), &gradient_output_tensor));

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
    const double regularizer = *regularizer_tensor.scalar<double>().data();
    const double torque_cost_weight = *gradient_torque_cost_tensor.scalar<double>().data();
    const double torque_limit_weight = *gradient_torque_limit_tensor.scalar<double>().data();
    const double velocity_limit_weight = *gradient_velocity_limit_tensor.scalar<double>().data();
    auto gradient_output = gradient_output_tensor->tensor<double, 3>();

    const double dt_squared = dt * dt;

    // B = batch size, T = length of the trajectory
    const auto B = trajectory_tensor.shape().dim_size(0);
    const auto T = trajectory_tensor.shape().dim_size(1);

    // pinocchio writes the results of the RNEA and its derivatives here.
    auto& tau = data.tau;
    auto& Jq = data.dtau_dq;
    auto& Jv = data.dtau_dv;
    auto& Ja = data.M;

    // Variables for calculating the Hessian (per batch element):
    Eigen::MatrixXd banded_hessian(3 * D, T * D);
    Eigen::MatrixXd torque_cost_jacobians((T + 1) * D, 3 * D);
    Eigen::ArrayXd torque_limit_mask_full((T + 1) * D);
    Eigen::ArrayXd velocity_limit_mask_full(T * D);

    // Variables for calculating the gradient (per batch element):
    // The gradients are calculated individually.
    Eigen::ArrayXd torque_cost_grad(T * D);
    Eigen::ArrayXd torque_limit_grad(T * D);
    Eigen::ArrayXd velocity_limit_grad(T * D);

    for(int64_t i = 0; i < B; ++i)
    {
      const auto* trajectory_ptr = trajectory.data() + i * T * D;
      for(int64_t j = 0; j < T; ++j)
      {
        Eigen::Map<const Eigen::VectorXd> q(j ? (trajectory_ptr + (j - 1) * D) : q0.data() + i * D, D);
        Eigen::Map<const Eigen::VectorXd> q_next(trajectory_ptr + j * D, D);
        Eigen::Map<const Eigen::VectorXd> q_prev(j >= 2 ? (trajectory_ptr + (j - 2) * D) : (j == 1 ? q0.data() + i * D : q_1.data() + i * D), D);
        const Eigen::VectorXd v = (q_next - q) / dt;
        const Eigen::VectorXd a = (q_next + q_prev - 2.0 * q) / dt_squared;
        std::lock_guard<std::mutex> lg(mutex);
        Jq.setZero();
        Jv.setZero();
        Ja.setZero();
        // computeRNEADerivatives must be called before rnea because it overwrites tau.
        pinocchio::computeRNEADerivatives(model, data, q, v, a);
        pinocchio::rnea(model, data, q, v, a);
        // pinocchio only fills the upper triangle:
        Ja.template triangularView<Eigen::StrictlyLower>() = Ja.transpose().template triangularView<Eigen::StrictlyLower>();
        Jv /= dt;
        if(acceleration_factors)
          Ja.diagonal() += Eigen::Map<const Eigen::VectorXd>(acceleration_factors, D);
        Ja /= dt_squared;

        // This is going to be used by the Hessian calculation:
        if(j >= 2)
          torque_cost_jacobians.block(j * D, 0, D, D) = Ja;
        if(j >= 1)
          torque_cost_jacobians.block(j * D, D, D, D) = -Jv + Jq - 2.0 * Ja;
        torque_cost_jacobians.block(j * D, 2 * D, D, D) = Jv + Ja;

        if(acceleration_factors)
          tau.array() += a.array() * Eigen::Map<const Eigen::ArrayXd>(acceleration_factors, D);
        tau.array() /= max_torques;

        // For the Jacobian of the torque limit loss, we need the Jacobians before the multiplication by tau.
        auto Jq_limit = Jq;
        auto Jv_limit = Jv;
        auto Ja_limit = Ja;
        Jq.array().colwise() *= tau.array() / max_torques;
        Jv.array().colwise() *= tau.array() / max_torques;
        Ja.array().colwise() *= tau.array() / max_torques;

        Eigen::ArrayXd torque_limit_mask(D);
        Eigen::ArrayXd velocity_limit_mask(D);
        for(int k = 0; k < D; ++k)
        {
          torque_limit_mask(k) = std::abs(tau(k)) > torque_limit_ratio ? (tau(k) > 0.0 ? 1.0 : -1.0) : 0.0;
          torque_limit_mask_full(j * D + k) = torque_limit_mask(k);
          velocity_limit_mask(k) = std::abs(v(k)) > velocity_limit_ratio * max_velocities(k) ? (v(k) > 0.0 ? 1.0 : -1.0) : 0.0;
          velocity_limit_mask_full(j * D + k) = velocity_limit_mask(k);
        }

        const auto tau_limit = (tau.array().abs() - torque_limit_ratio) * torque_limit_mask;
        Jq_limit.array().colwise() *= tau_limit / max_torques;
        Jv_limit.array().colwise() *= tau_limit / max_torques;
        Ja_limit.array().colwise() *= tau_limit / max_torques;

        if(j >= 2)
        {
          torque_cost_grad.segment((j - 2) * D, D) += Ja.colwise().sum().array();
          torque_limit_grad.segment((j - 2) * D, D) += Ja_limit.colwise().sum().array();
        }
        if(j >= 1)
        {
          torque_cost_grad.segment((j - 1) * D, D) += -Jv.colwise().sum().array() + Jq.colwise().sum().array() - 2.0 * Ja.colwise().sum().array();
          torque_limit_grad.segment((j - 1) * D, D) += -Jv_limit.colwise().sum().array() + Jq_limit.colwise().sum().array() - 2.0 * Ja_limit.colwise().sum().array();
          velocity_limit_grad.segment((j - 1) * D, D) -= (v.array().abs() / max_velocities - velocity_limit_ratio) / max_velocities * velocity_limit_mask;
        }
        // This line is always the first one that visits the gradient of a particular timestep, so we can overwrite what was there before.
        torque_cost_grad.segment(j * D, D) = Jv.colwise().sum().array() + Ja.colwise().sum().array();
        torque_limit_grad.segment(j * D, D) = Jv_limit.colwise().sum().array() + Ja_limit.colwise().sum().array();
        velocity_limit_grad.segment(j * D, D) = (v.array().abs() / max_velocities - velocity_limit_ratio) / max_velocities * velocity_limit_mask;
      }
      {
        Eigen::Map<const Eigen::VectorXd> q(trajectory_ptr + (T - 1) * D, D);
        const Eigen::VectorXd a = (Eigen::Map<const Eigen::VectorXd>(trajectory_ptr + (T - 2) * D, D) - q) / dt_squared;
        std::lock_guard<std::mutex> lg(mutex);
        Jq.setZero();
        Jv.setZero(); // We don't need Jv in this case, but who knows what pinocchio does with the values there?
        Ja.setZero();
        // computeRNEADerivatives must be called before rnea because it overwrites tau.
        pinocchio::computeRNEADerivatives(model, data, q, Eigen::VectorXd::Zero(D), a);
        pinocchio::rnea(model, data, q, Eigen::VectorXd::Zero(D), a);
        // pinocchio only fills the upper triangle:
        Ja.template triangularView<Eigen::StrictlyLower>() = Ja.transpose().template triangularView<Eigen::StrictlyLower>();
        if(acceleration_factors)
          Ja.diagonal() += Eigen::Map<const Eigen::VectorXd>(acceleration_factors, D);
        Ja /= dt_squared;

        // This is going to be used by the Hessian calculation:
        torque_cost_jacobians.block(T * D, 0, D, D) = Ja;
        torque_cost_jacobians.block(T * D, D, D, D) = Jq - Ja;

        if(acceleration_factors)
          tau.array() += a.array() * Eigen::Map<const Eigen::ArrayXd>(acceleration_factors, D);
        tau.array() /= max_torques;
        // tau must not be divided by max_torques^2 here because its original value is needed for comparing to the limits.

        // For the Jacobian of the torque limit loss, we need the Jacobians before the multiplication by tau.
        auto Jq_limit = Jq;
        auto Ja_limit = Ja;
        Jq.array().colwise() *= tau.array() / max_torques;
        Ja.array().colwise() *= tau.array() / max_torques;

        Eigen::ArrayXd torque_limit_mask(D);
        for(int k = 0; k < D; ++k)
        {
          torque_limit_mask(k) = std::abs(tau(k)) > torque_limit_ratio ? (tau(k) > 0.0 ? 1.0 : -1.0) : 0.0;
          torque_limit_mask_full(T * D + k) = torque_limit_mask(k);
        }

        const auto tau_limit = (tau.array().abs() - torque_limit_ratio) * torque_limit_mask;
        Jq_limit.array().colwise() *= tau_limit / max_torques;
        Ja_limit.array().colwise() *= tau_limit / max_torques;

        torque_cost_grad.segment((T - 2) * D, D) += Ja.colwise().sum().array();
        torque_limit_grad.segment((T - 2) * D, D) += Ja_limit.colwise().sum().array();

        torque_cost_grad.segment((T - 1) * D, D) += Jq.colwise().sum().array() - Ja.colwise().sum().array();
        torque_limit_grad.segment((T - 1) * D, D) += Jq_limit.colwise().sum().array() - Ja_limit.colwise().sum().array();
      }

      // Incorporate normalization factors such that being at the limit equals 1.
      // Velocity limit gradient also gets a 1/dt that was factored out.
      torque_limit_grad /= (1.0 - torque_limit_ratio) * (1.0 - torque_limit_ratio);
      velocity_limit_grad /= dt * (1.0 - velocity_limit_ratio) * (1.0 - velocity_limit_ratio);

      // Combine the gradient from weighted components.
      Eigen::Map<Eigen::ArrayXd>(gradient_output.data() + i * T * D, T * D) = (torque_cost_weight * torque_cost_grad + torque_limit_weight * torque_limit_grad + velocity_limit_weight * velocity_limit_grad) / B;
      // If present, add the gradient from other loss components.
      if(gradient_trajectory_tensor.shape().dims() > 0)
        Eigen::Map<Eigen::ArrayXd>(gradient_output.data() + i * T * D, T * D) += Eigen::Map<const Eigen::ArrayXd>(gradient_trajectory_tensor.tensor<double, 3>().data() + i * T * D, T * D);

      // Calculate the Hessian.
      for(int64_t j = 0; j < 3 * D; ++j)
      {
        // j is the "off-diagonality".
        for(int64_t k = 0; k < T * D - j; ++k)
        {
          // k counts along the (off-)diagonal.
          // All of those are dot products of columns of the (not in memory existing) Jacobian of the entire residual.
          // Namely this is the dot product of columns k and (k + j).
          double x = 0.0;
          const auto start_block = (k + j) / D;
          const auto end_block = std::min(k / D + 3, T + 1);
          for(auto idx = start_block; idx < end_block; ++idx)
            for(auto l = decltype(D){0}; l < D; ++l)
            {
              x += (torque_cost_weight + torque_limit_weight * torque_limit_mask_full(idx * D + l) * torque_limit_mask_full(idx * D + l) / ((1.0 - torque_limit_ratio) * (1.0 - torque_limit_ratio))) * torque_cost_jacobians(idx * D + l, k + j - (idx - 2) * D) * torque_cost_jacobians(idx * D + l, k - (idx - 2) * D) / (max_torques(l) * max_torques(l));
            }
          if(j == 0)
          {
            const auto l = k % D;
            x += velocity_limit_weight * velocity_limit_mask_full(k) * velocity_limit_mask_full(k) / (dt * dt * (1.0 - velocity_limit_ratio) * (1.0 - velocity_limit_ratio) * max_velocities(l) * max_velocities(l));
            if(k / D < T - 1)
              x += velocity_limit_weight * velocity_limit_mask_full(k + D) * velocity_limit_mask_full(k + D) / (dt * dt * (1.0 - velocity_limit_ratio) * (1.0 - velocity_limit_ratio) * max_velocities(l) * max_velocities(l));
          }
          else if(j == D)
          {
            const auto l = k % D;
            x -= velocity_limit_weight * velocity_limit_mask_full(k + j) * velocity_limit_mask_full(k + j) / (dt * dt * (1.0 - velocity_limit_ratio) * (1.0 - velocity_limit_ratio) * max_velocities(l) * max_velocities(l));
          }
          banded_hessian(j, k) = x / B;
        }
      }

      // Add the regularizer to the diagonal of the Hessian (which, due to the banded representation, is the first row).
      banded_hessian.row(0).array() += regularizer;

      // Let LAPACK solve the system (in place).
      const char uplo = 'L';
      const int n = T * D;
      const int kd = 3 * D - 1;
      const int nrhs = 1;
      const int ldab = 3 * D;
      const int ldb = n;
      int info = 0;
      dpbsv_(&uplo, &n, &kd, &nrhs, banded_hessian.data(), &ldab, gradient_output.data() + i * T * D, &ldb, &info);
      if(info != 0)
        fprintf(stderr, "LAPACK error when decomposing the Hessian: %d\n", info);
    }
  }

private:
  pinocchio::Model model; ///< The robot model.
  pinocchio::Data data; ///< The data on which pinocchio operates.
  std::mutex mutex; ///< Mutex to synchronize access to \c data (the TF docs say that \c Compute can be called from multiple threads simultaneously).
};

REGISTER_KERNEL_BUILDER(Name("DynamicLossGradient").Device(DEVICE_CPU), DynamicLossGradientOp);
