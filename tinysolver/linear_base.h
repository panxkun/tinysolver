#ifndef __OPTIMIZER_LINEAR_BASE_H__
#define __OPTIMIZER_LINEAR_BASE_H__

#include <ceres/ceres.h>
#include <slamtool/common.h>
#include <slamtool/estimation/tinysolver/tiny_reprojection_factor.h>
#include <slamtool/estimation/tinysolver/tiny_preintegration_factor.h>
#include <slamtool/estimation/tinysolver/tiny_marginalization_factor.h>
#include <slamtool/estimation/tinysolver/loss_function.h>

namespace slamtool{

enum ParamType{
    PARAM_CAMERA,
    PARAM_CAMERA_CONST,
    PARAM_LANDMARK,
    PARAM_LANDMARK_CONST,   
    PARAM_MOTION_BIAS_GYR,
    PARAM_MOTION_BIAS_GYR_CONST,
    PARAM_MOTION_BIAS_ACC,
    PARAM_MOTION_BIAS_ACC_CONST,
    PARAM_MOTION_VELOCITY,
    PARAM_MOTION_VELOCITY_CONST
};

class ResidualBlockBase;
class VisualResidualBlock;
class MotionResidualBlock;
class ParameterBlock;
class CameraBundle;
class LandmarkBlock;

using ResidualBlockId           = ResidualBlockBase*;
using VisualResidualBlockId     = VisualResidualBlock*;
using MotionResidualBlockId     = MotionResidualBlock*;
using ParameterBlockId          = ParameterBlock*;
using CameraBundleId            = CameraBundle*;
using LandmarkId                = LandmarkBlock*;

using SparseBlockStorage = std::vector<std::map<int, matrix<15, 15>>>;

class Corrector{
public:
    double sqrt_rho1_;
    double residual_scaling_;
    double alpha_sq_norm_;

    Corrector(const double sq_norm, const double rho[3]){
        sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
            return;
        }

        const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        const double alpha = 1.0 - sqrt(D);

        residual_scaling_ = sqrt_rho1_ / (1 - alpha);
        alpha_sq_norm_ = alpha / sq_norm;
    }

    template <int R>
    void CorrectResidual(double *res_ptr){
        map<vector<R, true>> residuals(res_ptr);
        residuals *= residual_scaling_;
    }

    template <int R, int C>
    void CorrectJacobain(double *res_ptr, double *jac_ptr){
        map<vector<R, true>> residuals(res_ptr);
        map<matrix<R, C, true>> jacobian(jac_ptr);

        if(alpha_sq_norm_ == 0.0){
            jacobian *= sqrt_rho1_;
            return;
        }

        jacobian = sqrt_rho1_ * (jacobian - alpha_sq_norm_ * residuals * residuals.transpose() * jacobian);
    }
};

struct ResidualBlockIndex{
    ResidualBlockId residual_block;
    size_t index;
};

class ParameterBlock{

public:
    size_t id = 0;
    size_t size;
    double *param_ptr_raw;
    std::vector<double> param_cpy;
    std::vector<double> param_new;
    ParamType type;
    
    std::shared_ptr<ceres::LocalParameterization> parameterization_ptr;
    ceres::LocalParameterization *parameterization;

    ParameterBlock(double *_ptr, size_t _size, ParamType _type, ceres::LocalParameterization *_p):
    param_ptr_raw(_ptr), size(_size), type(_type), parameterization(_p){
        param_cpy.resize(size);
        param_new.resize(size);
        for(size_t i = 0; i < size; i++){
            param_cpy[i] = param_ptr_raw[i];
            param_new[i] = param_ptr_raw[i];
        }

        if(!parameterization){
            parameterization_ptr = std::make_shared<ceres::IdentityParameterization>(size);
            parameterization = parameterization_ptr.get();
        }
    }

    void Update() {
        param_cpy = param_new;
    }

    void PlusDelta(const double *delta){
        parameterization->Plus(param_cpy.data(), delta, param_new.data());
    }

    double StepSquareNorm(){
        double sn = 0.0;
        for(size_t i = 0; i < size; i++)
            sn += (param_new[i] - param_cpy[i]) * (param_new[i] - param_cpy[i]);
        return sn;
    }

    double XSquareNorm(){
        double sn = 0.0;
        for(size_t i = 0; i < size; i++)
            sn += param_cpy[i] * param_cpy[i];
        return sn;
    }

    void AcceptResult(){
        for(size_t i = 0; i < size; i++)
            param_ptr_raw[i] = param_cpy[i];
    }
};

class ResidualBlockBase{
public:
    size_t id = 0;
    LossFunction* loss_func = nullptr;
    virtual ~ResidualBlockBase() = default;
    virtual double ComputeCost() = 0;
    virtual double ComputeCost(vector<> res) = 0;
    virtual void CheckConstantParam() = 0;
    virtual void PreComputeBlock() = 0;
};

struct CameraBundle{
    size_t id = 0;
    ParamType type;
    ParameterBlockId frame_p = nullptr;
    ParameterBlockId frame_q = nullptr;
    ParameterBlockId velocity = nullptr;
    ParameterBlockId bias_gyr = nullptr;
    ParameterBlockId bias_acc = nullptr;

    CameraBundle(ParameterBlockId _frame_p, ParameterBlockId _frame_q, ParamType type): 
    frame_p(_frame_p), frame_q(_frame_q), type(type){
        velocity = nullptr;
        bias_gyr = nullptr;
        bias_acc = nullptr;
    }

    std::vector<LandmarkId> landmarks;
    std::vector<VisualResidualBlockId> visual_residual_blocks;
    std::vector<MotionResidualBlockId> motion_residual_blocks;
};

struct LandmarkBlock{
    size_t id = 0;
    ParamType type;
    ParameterBlockId landmark;

    LandmarkBlock(ParameterBlockId _landmark, ParamType type):landmark(_landmark), type(type){}

    std::vector<std::pair<CameraBundleId, CameraBundleId>> camera_bundles;
    std::vector<VisualResidualBlockId> residual_blocks;
};

class MotionResidualBlock: public ResidualBlockBase{

public:
    TinyPreIntegrationErrorFactor* factor;

    CameraBundleId camera_bundle_tgt;
    CameraBundleId camera_bundle_ref;

    std::array<double*, 10> jacobian_ptr;
    std::array<double*, 10> param_block_ptr;
    std::array<double*, 10> param_block_candidate_ptr;

    matrix<15, 4, true> jacobian_tgt_q;
    matrix<15, 3, true> jacobian_tgt_p;
    matrix<15, 3, true> jacobian_tgt_v;
    matrix<15, 3, true> jacobian_tgt_bg;
    matrix<15, 3, true> jacobian_tgt_ba;
    matrix<15, 4, true> jacobian_ref_q;
    matrix<15, 3, true> jacobian_ref_p;
    matrix<15, 3, true> jacobian_ref_v;
    matrix<15, 3, true> jacobian_ref_bg;
    matrix<15, 3, true> jacobian_ref_ba;

    std::array<matrix<15, 6>, 2> jacobian_camera;
    std::array<matrix<15, 9>, 2> jacobian_motion;

    matrix<15, 1> residual;
    matrix<15, 1> residual_candidate;

    MotionResidualBlock(TinyPreIntegrationErrorFactor* factor, LossFunction* loss_function): factor(factor){
        loss_func = loss_function;
    }

    double ComputeCost() override {
        return loss_func->Compute(residual.squaredNorm());
    }
    
    double ComputeCost(vector<> res) override {
        return loss_func->Compute(res.squaredNorm());
    }

    void PreComputeBlock() override {
        
        jacobian_camera[0] << jacobian_tgt_q.leftCols(3), jacobian_tgt_p;
        jacobian_camera[1] << jacobian_ref_q.leftCols(3), jacobian_ref_p;

        jacobian_motion[0] << jacobian_tgt_v, jacobian_tgt_bg, jacobian_tgt_ba;
        jacobian_motion[1] << jacobian_ref_v, jacobian_ref_bg, jacobian_ref_ba;
    }

    void CheckConstantParam() override {
        if(camera_bundle_tgt->type == PARAM_CAMERA_CONST){
            jacobian_tgt_q.setZero();
            jacobian_tgt_p.setZero();
        }
        if(camera_bundle_ref->type == PARAM_CAMERA_CONST){
            jacobian_ref_q.setZero();
            jacobian_ref_p.setZero();
        }
        if(camera_bundle_tgt->velocity->type == PARAM_MOTION_VELOCITY_CONST){
            jacobian_tgt_v.setZero();
        }
        if(camera_bundle_tgt->bias_gyr->type == PARAM_MOTION_BIAS_GYR_CONST){
            jacobian_tgt_bg.setZero();
        }
        if(camera_bundle_tgt->bias_acc->type == PARAM_MOTION_BIAS_ACC_CONST){
            jacobian_tgt_ba.setZero();
        }
        if(camera_bundle_ref->velocity->type == PARAM_MOTION_VELOCITY_CONST){
            jacobian_ref_v.setZero();
        }
        if(camera_bundle_ref->bias_gyr->type == PARAM_MOTION_BIAS_GYR_CONST){
            jacobian_ref_bg.setZero();
        }
        if(camera_bundle_ref->bias_acc->type == PARAM_MOTION_BIAS_ACC_CONST){
            jacobian_ref_ba.setZero();
        }
    }
};

class VisualResidualBlock: public ResidualBlockBase{

public:

    TinyReprojectionErrorFactor* factor;

    // important to use RowMajor
    matrix<2, 4, true> jacobian_ref_q;
    matrix<2, 3, true> jacobian_ref_p;
    matrix<2, 4, true> jacobian_tgt_q;
    matrix<2, 3, true> jacobian_tgt_p;
    matrix<2, 1, true> jacobian_inv_depth;

    matrix<2, 1> residual;
    matrix<2, 1> residual_candidate;

    // J = [F, E]
    matrix1 ete;
    vector1 etb;
    std::array<matrix6x1, 2> etf;
    std::array<matrix6, 4> ftf;
    std::array<vector6, 2> ftb;
    std::array<matrix<2, 6>, 2> jacobian_camera;

    LandmarkId landmark;
    CameraBundleId camera_bundle_tgt;
    CameraBundleId camera_bundle_ref;

    std::array<double*, 5> jacobian_ptr;
    std::array<double*, 5> param_block_ptr;
    std::array<double*, 5> param_block_candidate_ptr;

    VisualResidualBlock(TinyReprojectionErrorFactor* factor, LossFunction* loss_function): factor(factor){
        loss_func = loss_function;
    }

    double ComputeCost() override {
        return loss_func->Compute(residual.squaredNorm());
    }
    
    double ComputeCost(vector<> res) override {
        return loss_func->Compute(res.squaredNorm());
    }

    void PreComputeBlock() override {

        double rho[3];
        vector<2> res = residual;
        loss_func->Evaluate(res.squaredNorm(), rho);

        jacobian_camera[0] << jacobian_tgt_q.leftCols(3), jacobian_tgt_p;
        jacobian_camera[1] << jacobian_ref_q.leftCols(3), jacobian_ref_p;

        Corrector corrector(res.squaredNorm(), rho);
        corrector.CorrectJacobain<2, 6>(res.data(), jacobian_camera[0].data());
        corrector.CorrectJacobain<2, 6>(res.data(), jacobian_camera[1].data());
        corrector.CorrectJacobain<2, 1>(res.data(), jacobian_inv_depth.data());
        corrector.CorrectResidual<2>(res.data());

        ete.noalias() = jacobian_inv_depth.transpose() * jacobian_inv_depth;
        etb.noalias() = jacobian_inv_depth.transpose() * res * -1.0;

        etf[0].noalias() = jacobian_camera[0].transpose() * jacobian_inv_depth;
        etf[1].noalias() = jacobian_camera[1].transpose() * jacobian_inv_depth;

        ftf[0].noalias() = jacobian_camera[0].transpose() * jacobian_camera[0];
        ftf[1].noalias() = jacobian_camera[0].transpose() * jacobian_camera[1];
        ftf[2].noalias() = jacobian_camera[1].transpose() * jacobian_camera[0];
        ftf[3].noalias() = jacobian_camera[1].transpose() * jacobian_camera[1];

        ftb[0].noalias() = jacobian_camera[0].transpose() * res * -1.0;
        ftb[1].noalias() = jacobian_camera[1].transpose() * res * -1.0;
    }

    void CheckConstantParam() override {
        if(camera_bundle_tgt->type == PARAM_CAMERA_CONST){
            jacobian_tgt_q.setZero();
            jacobian_tgt_p.setZero();
        }
        if(camera_bundle_ref->type == PARAM_CAMERA_CONST){
            jacobian_ref_q.setZero();
            jacobian_ref_p.setZero();
        }
        if(landmark->type == PARAM_LANDMARK_CONST){
            jacobian_inv_depth.setZero();
        }
    }
};

class MarginResidualBlock: public ResidualBlockBase{

public:

    TinyMarginalizationFactor*      factor;
    std::vector<CameraBundleId>     camera_bundles;
    std::vector<double*>            jacobians_ptr;
    std::vector<double*>            param_block_ptr;
    std::vector<double*>            param_block_candidate_ptr;
    std::vector<matrix<>>           jacobians_state;
    vector<>                        residuals;
    vector<>                        residuals_candidate;
    matrix<>                        jacobians_matrix;

    std::vector<matrix<Eigen::Dynamic, Eigen::Dynamic, true>>   jacobians;

    MarginResidualBlock(TinyMarginalizationFactor* factor, LossFunction* loss_function): factor(factor){
        const auto frames = factor->linearization_frames();
        param_block_ptr.resize(frames.size() * 5);
        param_block_candidate_ptr.resize(frames.size() * 5);
        jacobians_ptr.resize(frames.size() * 5);
        jacobians.resize(frames.size() * 5);
        jacobians_state.resize(frames.size());
        residuals.resize(frames.size() * ES_SIZE);
        residuals_candidate.resize(frames.size() * ES_SIZE);
        jacobians_matrix.resize(frames.size() * ES_SIZE, frames.size() * ES_SIZE);

        loss_func = loss_function;
    }

    double ComputeCost() override {
        return loss_func->Compute(residuals.squaredNorm());
    }

    double ComputeCost(vector<> res) override {
        return loss_func->Compute(res.squaredNorm());
    }

    void PreComputeBlock(){
        const auto &frames = factor->linearization_frames();
        for(size_t i = 0; i < frames.size(); i++){
            matrix<>& dr_ds = jacobians_state[i];
            dr_ds.resize(frames.size() * ES_SIZE, ES_SIZE);
            dr_ds.block(0, ES_Q, frames.size() * ES_SIZE, 3) = jacobians[i * 5 + 0].leftCols(3);
            dr_ds.block(0, ES_P, frames.size() * ES_SIZE, 3) = jacobians[i * 5 + 1];
            dr_ds.block(0, ES_V, frames.size() * ES_SIZE, 3) = jacobians[i * 5 + 2];
            dr_ds.block(0, ES_BG, frames.size() * ES_SIZE, 3) = jacobians[i * 5 + 3];
            dr_ds.block(0, ES_BA, frames.size() * ES_SIZE, 3) = jacobians[i * 5 + 4];
            jacobians_matrix.block(0, i * ES_SIZE, frames.size() * ES_SIZE, ES_SIZE) = dr_ds;
        }
    }

    void CheckConstantParam() override {

    }
};

}

#endif