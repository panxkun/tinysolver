#ifndef __OPTIMIZER_LINEAR_PROBLEM_H__
#define __OPTIMIZER_LINEAR_PROBLEM_H__

#include <slamtool/common.h>
#include <slamtool/estimation/state.h>
#include <ceres/ceres.h>
#include <slamtool/estimation/tinysolver/loss_function.h>
#include <slamtool/estimation/tinysolver/tiny_reprojection_factor.h>
#include <slamtool/estimation/tinysolver/tiny_preintegration_factor.h>
#include <slamtool/estimation/tinysolver/tiny_marginalization_factor.h>
#include <slamtool/estimation/tinysolver/linear_base.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace ceres::internal;

// #define CFG_OPTIMIZER_DEBUG
// #define CFG_TIME_EVALUATE

#ifdef CFG_TIME_EVALUATE
    #define TIME_EVALUATE(func)                                             \
    {                                                                       \
        auto start = std::chrono::high_resolution_clock::now();             \
        func;                                                               \
        auto end = std::chrono::high_resolution_clock::now();               \
        std::chrono::duration<double, std::milli> duration = end - start;   \
        log_debug("Execution time of %s: %f ms", #func, duration.count());  \
    }
#else
    #define TIME_EVALUATE(func) func
#endif

namespace slamtool{

enum BundleAdjustmentBlockDIM {
    BA_V_DIM = 6,
    BA_M_DIM = 9,
    BA_S_DIM = 15,
    BA_V_RES = 2,
    BA_M_RES = 15,
    BA_S_RES = 15
};

class LinearProblem{

public:
    double  _radius                         = 1.0e4;
    double  _decrease_factor                = 2.0;
    double  _current_cost                   = 0.0; 
    size_t  _num_invalid_steps              = 0;
    double  _solver_start_wall_time         = 0.0;
    bool    _visual_constraint              = false;
    bool    _motion_constraint              = false;
    bool    _margin_constraint              = false;
    bool    _is_succeed                     = false;
    bool    _reuse_last_state               = false;

    Timer _timer;
    double GetWallTimeInSeconds() {return _timer.get_time();}
    double GetWallTimeInMilliSeconds() {return GetWallTimeInSeconds() * 1000;}
    double GetWallTimeInMicroSeconds() {return GetWallTimeInSeconds() * 1000000;}

    struct IterationSummary{
        size_t iteration                    = 0;
        double model_cost_change            = 0.0;
        double cost                         = 0.0;
        double candidate_cost               = 0.0;
        double cost_change                  = 0.0;
        double previous_cost                = 0.0;
        double relative_decrease            = 0.0;
        double gradient_max_norm            = 0.0;
        double gradient_norm                = 0.0;
        double step_norm                    = 0.0;
        double tr_ratio                     = 0.0;
        double tr_radius                    = 0.0;
        double iteration_start_wall_time    = 0.0;
        bool step_is_valid                  = false;
    };

    IterationSummary iteration_summary;

    struct Options{
        bool verbose                        = false;
        size_t num_threads                  = 1;
        size_t max_num_invalid_steps        = 3;
        double radius                       = 1.0e4;
        double min_radius                   = 1.0e-32;
        double max_radius                   = 1.0e+16;
        double decrease_factor              = 2.0;
        double function_tolerance           = 1.0e-6;
        double parameter_tolerance          = 1.0e-8;
        double gradient_tolerance           = 1.0e-10;
        double max_solver_time_in_seconds   = 1.0e6;
        double max_num_iterations           = 10;
        double min_relative_decrease        = 1.0e-3;
    };

    Options options;

    std::vector<std::unique_ptr<ParameterBlock>>            param_blocks;
    std::vector<std::unique_ptr<VisualResidualBlock>>       visual_residual_blocks;
    std::vector<std::unique_ptr<MotionResidualBlock>>       motion_residual_blocks;
    std::vector<std::unique_ptr<MarginResidualBlock>>       margin_residual_blocks;
    std::vector<std::unique_ptr<CameraBundle>>              camera_bundles;
    std::vector<std::unique_ptr<LandmarkBlock>>             landmarks;

    std::unordered_map<const double*, ParameterBlockId>     pointer_to_param_id_map;
    std::unordered_map<const double*, CameraBundleId>       pointer_to_camera_bundle_id_map;
    std::unordered_map<const double*, LandmarkId>           pointer_to_landmark_id_map;

    struct BundleAdjustBlocks{

        vector<>                                    sc_states;
        matrix<>                                    sc_matrix;
        vector<>                                    sc_vector;
        std::vector<matrix1>                        sc_etes;
        std::vector<matrix1>                        sc_etes_inv;
        std::vector<vector1>                        sc_etbs;
        std::array<std::vector<matrix<6, 1>>, 2>    sc_etfs;

        matrix<> visual_matrix;
        vector<> visual_vector;
        matrix<> motion_matrix;
        vector<> motion_vector;
        matrix<> margin_matrix;
        vector<> margin_vector;

        vector<> state_pose;
        vector<> state_motion;
        vector<> state_landmark;

#ifdef CFG_OPTIMIZER_DEBUG
        matrix<> visual_jacobian;
        vector<> visual_residual;
        matrix<> visual_hessian;
        matrix<> visual_vector_sc;
        matrix<> visual_matrix_sc;
        matrix<> visual_ftf;
        matrix<> visual_ete;
        matrix<> visual_etf;

        matrix<> motion_jacobian;
        vector<> motion_residual;
        matrix<> motion_hessian;

        matrix<> visual_motion_jacobian;
        vector<> visual_motion_residual;
        matrix<> visual_motion_hessian;
        matrix<> visual_motion_vector_sc;
        matrix<> visual_motion_matrix_sc;
        matrix<> visual_motion_ftf;
        matrix<> visual_motion_ete;
        matrix<> visual_motion_etf;        

        matrix<> margin_jacobian;
        vector<> margin_residual;
        matrix<> margin_hessian;
#endif
        void reset(bool reuse=false){

            sc_matrix.setZero();
            sc_vector.setZero();
            sc_states.setZero();
            for(auto m: sc_etfs[0])     m.setZero();
            for(auto m: sc_etfs[1])     m.setZero();
            for(auto m: sc_etes)        m.setZero();
            for(auto m: sc_etes_inv)    m.setZero();
            for(auto m: sc_etbs)        m.setZero();

            if(reuse) return;

            visual_matrix.setZero();
            visual_vector.setZero();
            motion_matrix.setZero();
            motion_vector.setZero();
            margin_matrix.setZero();
            margin_vector.setZero();

            state_pose.setZero();
            state_motion.setZero();
            state_landmark.setZero();

#ifdef CFG_OPTIMIZER_DEBUG
            visual_jacobian.setZero();
            visual_residual.setZero();
            visual_hessian.setZero();
            visual_ftf.setZero();
            visual_ete.setZero();
            visual_etf.setZero();

            motion_jacobian.setZero();
            motion_residual.setZero();
            motion_hessian.setZero();

            margin_jacobian.setZero();
            margin_residual.setZero();
            margin_hessian.setZero();
#endif
        }
    };

    std::unique_ptr<BundleAdjustBlocks> ba_block;

    size_t num_landmarks        = 0;
    size_t num_camera_bundles   = 0;
    size_t num_visual_residuals = 0;
    size_t num_motion_residuals = 0;
    size_t num_margin_residuals = 0;

public:

    LinearProblem(){
        _timer = Timer();
        _solver_start_wall_time = GetWallTimeInSeconds();
    }

    ~LinearProblem() = default;

    bool Solve(Options options){

        this->options = options;
        
        PreprocessData();

        IterationZero();

        while(CheckTerminationCriteriaAndLog()){

            if(ComputeCandidatePointAndEvaluateCost()){
                HandSuccessfulStep();
            }else{
                HandInvalidStep();
                continue;
            }

            if(ParameterToleranceReached()) 
                return false;

            if(FunctionToleranceReached()) 
                return false;
        }

        if(_is_succeed){
            AcceptOptimizationResult();
            return true;
        }
        return false;
    }

    ParameterBlockId AddParameterBlock(double* values, int size, ParamType type, ceres::LocalParameterization* local_parameterization=nullptr){
        //TODO: maybe need to check the parameterization type in deifferent add process
        const auto it = pointer_to_param_id_map.find(values);
        if(it == pointer_to_param_id_map.end()){
            auto pb = std::make_unique<ParameterBlock>(values, size, type, local_parameterization);
            param_blocks.push_back(std::move(pb));
            pointer_to_param_id_map[values] = param_blocks.back().get();
        }
        return pointer_to_param_id_map[values];
    }

    ResidualBlockId AddResidualBlock(TinyReprojectionErrorFactor* cost_function, LossFunction* loss_function){
        
        auto tgt_frame = cost_function->frame;
        auto ref_frame = cost_function->track->first_frame();
        auto track     = cost_function->track;

        ParameterBlockId tgt_frame_q_pb     = AddParameterBlock(tgt_frame->pose.q.coeffs().data(), 4, PARAM_CAMERA); // need local parameterization
        ParameterBlockId tgt_frame_p_pb     = AddParameterBlock(tgt_frame->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId ref_frame_q_pb     = AddParameterBlock(ref_frame->pose.q.coeffs().data(), 4, PARAM_CAMERA);
        ParameterBlockId ref_frame_p_pb     = AddParameterBlock(ref_frame->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId landmark_pb        = AddParameterBlock(&track->landmark.inv_depth, 1, PARAM_LANDMARK);
        LandmarkId lm                       = AddLandmark(landmark_pb);
        CameraBundleId cb_tgt               = AddCameraBundle(tgt_frame_p_pb, tgt_frame_q_pb);
        CameraBundleId cb_ref               = AddCameraBundle(ref_frame_p_pb, ref_frame_q_pb);

        std::unique_ptr<VisualResidualBlock> rb = std::make_unique<VisualResidualBlock>(cost_function, loss_function);
        rb->landmark = lm;
        rb->camera_bundle_tgt = cb_tgt;
        rb->camera_bundle_ref = cb_ref;

        rb->jacobian_ptr = {
            rb->jacobian_tgt_q.data(),
            rb->jacobian_tgt_p.data(),
            rb->jacobian_ref_q.data(),
            rb->jacobian_ref_p.data(),
            rb->jacobian_inv_depth.data()
        };
        rb->param_block_ptr = {
            rb->camera_bundle_tgt->frame_q->param_cpy.data(),
            rb->camera_bundle_tgt->frame_p->param_cpy.data(),
            rb->camera_bundle_ref->frame_q->param_cpy.data(),
            rb->camera_bundle_ref->frame_p->param_cpy.data(),
            rb->landmark->landmark->param_cpy.data()
        };
        rb->param_block_candidate_ptr = {
            rb->camera_bundle_tgt->frame_q->param_new.data(),
            rb->camera_bundle_tgt->frame_p->param_new.data(),
            rb->camera_bundle_ref->frame_q->param_new.data(),
            rb->camera_bundle_ref->frame_p->param_new.data(),
            rb->landmark->landmark->param_new.data()
        };

        rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
        rb->PreComputeBlock();

        visual_residual_blocks.push_back(std::move(rb));

        lm->camera_bundles.push_back(std::make_pair(cb_tgt, cb_ref));
        lm->residual_blocks.push_back(visual_residual_blocks.back().get());
        cb_tgt->landmarks.push_back(lm);
        cb_tgt->visual_residual_blocks.push_back(visual_residual_blocks.back().get());
        cb_ref->landmarks.push_back(lm);
        cb_ref->visual_residual_blocks.push_back(visual_residual_blocks.back().get());

        return visual_residual_blocks.back().get();
    }

    ResidualBlockId AddResidualBlock(TinyReprojectionPriorFactor* cost_function, LossFunction* loss_function){
        
        auto tgt_frame = cost_function->rpefactor.frame;
        auto ref_frame = cost_function->rpefactor.track->first_frame();
        auto track     = cost_function->rpefactor.track;

        ParameterBlockId tgt_frame_q_pb     = AddParameterBlock(tgt_frame->pose.q.coeffs().data(), 4, PARAM_CAMERA); // need local parameterization
        ParameterBlockId tgt_frame_p_pb     = AddParameterBlock(tgt_frame->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId ref_frame_q_pb     = AddParameterBlock(ref_frame->pose.q.coeffs().data(), 4, PARAM_CAMERA_CONST);
        ParameterBlockId ref_frame_p_pb     = AddParameterBlock(ref_frame->pose.p.data(), 3, PARAM_CAMERA_CONST);
        ParameterBlockId landmark_pb        = AddParameterBlock(&track->landmark.inv_depth, 1, PARAM_LANDMARK_CONST);
        LandmarkId lm                       = AddLandmark(landmark_pb);
        CameraBundleId cb_tgt               = AddCameraBundle(tgt_frame_p_pb, tgt_frame_q_pb);
        CameraBundleId cb_ref               = AddCameraBundle(ref_frame_p_pb, ref_frame_q_pb);

        std::unique_ptr<VisualResidualBlock> rb = std::make_unique<VisualResidualBlock>(&(cost_function->rpefactor), loss_function);
        rb->landmark = lm;
        rb->camera_bundle_tgt = cb_tgt;
        rb->camera_bundle_ref = cb_ref;

        rb->jacobian_ptr = {
            rb->jacobian_tgt_q.data(),
            rb->jacobian_tgt_p.data(),
            rb->jacobian_ref_q.data(),
            rb->jacobian_ref_p.data(),
            rb->jacobian_inv_depth.data()
        };
        rb->param_block_ptr = {
            rb->camera_bundle_tgt->frame_q->param_cpy.data(),
            rb->camera_bundle_tgt->frame_p->param_cpy.data(),
            rb->camera_bundle_ref->frame_q->param_cpy.data(),
            rb->camera_bundle_ref->frame_p->param_cpy.data(),
            rb->landmark->landmark->param_cpy.data()
        };
        rb->param_block_candidate_ptr = {
            rb->camera_bundle_tgt->frame_q->param_new.data(),
            rb->camera_bundle_tgt->frame_p->param_new.data(),
            rb->camera_bundle_ref->frame_q->param_new.data(),
            rb->camera_bundle_ref->frame_p->param_new.data(),
            rb->landmark->landmark->param_new.data()
        };

        rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
        rb->PreComputeBlock();

        visual_residual_blocks.push_back(std::move(rb));

        lm->camera_bundles.push_back(std::make_pair(cb_tgt, cb_ref));
        lm->residual_blocks.push_back(visual_residual_blocks.back().get());
        cb_tgt->landmarks.push_back(lm);
        cb_tgt->visual_residual_blocks.push_back(visual_residual_blocks.back().get());
        cb_ref->landmarks.push_back(lm);
        cb_ref->visual_residual_blocks.push_back(visual_residual_blocks.back().get());

        return visual_residual_blocks.back().get();
    }

    ResidualBlockId AddResidualBlock(TinyPreIntegrationErrorFactor* cost_function, LossFunction* loss_function){
        
        auto frame_i = cost_function->frame_i;
        auto frame_j = cost_function->frame_j;

        ParameterBlockId tgt_frame_q_pb     = AddParameterBlock(frame_i->pose.q.coeffs().data(), 4, PARAM_CAMERA); // need local parameterization
        ParameterBlockId tgt_frame_p_pb     = AddParameterBlock(frame_i->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId tgt_frame_v_pb     = AddParameterBlock(frame_i->motion.v.data(), 3, PARAM_MOTION_VELOCITY);
        ParameterBlockId tgt_frame_bg_pb    = AddParameterBlock(frame_i->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR);
        ParameterBlockId tgt_frame_ba_pb    = AddParameterBlock(frame_i->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC);
        ParameterBlockId ref_frame_q_pb     = AddParameterBlock(frame_j->pose.q.coeffs().data(), 4, PARAM_CAMERA);
        ParameterBlockId ref_frame_p_pb     = AddParameterBlock(frame_j->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId ref_frame_v_pb     = AddParameterBlock(frame_j->motion.v.data(), 3, PARAM_MOTION_VELOCITY);
        ParameterBlockId ref_frame_bg_pb    = AddParameterBlock(frame_j->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR);
        ParameterBlockId ref_frame_ba_pb    = AddParameterBlock(frame_j->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC);
        CameraBundleId cb_tgt               = AddCameraBundle(tgt_frame_p_pb, tgt_frame_q_pb, tgt_frame_v_pb, tgt_frame_bg_pb, tgt_frame_ba_pb);
        CameraBundleId cb_ref               = AddCameraBundle(ref_frame_p_pb, ref_frame_q_pb, ref_frame_v_pb, ref_frame_bg_pb, ref_frame_ba_pb);

        std::unique_ptr<MotionResidualBlock> rb = std::make_unique<MotionResidualBlock>(cost_function, loss_function);
        rb->camera_bundle_tgt = cb_tgt;
        rb->camera_bundle_ref = cb_ref;

        rb->jacobian_ptr = {
            rb->jacobian_tgt_q.data(),
            rb->jacobian_tgt_p.data(),
            rb->jacobian_tgt_v.data(),
            rb->jacobian_tgt_bg.data(),
            rb->jacobian_tgt_ba.data(),
            rb->jacobian_ref_q.data(),
            rb->jacobian_ref_p.data(),
            rb->jacobian_ref_v.data(),
            rb->jacobian_ref_bg.data(),
            rb->jacobian_ref_ba.data()
        };
        rb->param_block_ptr = {
            rb->camera_bundle_tgt->frame_q->param_cpy.data(),
            rb->camera_bundle_tgt->frame_p->param_cpy.data(),
            rb->camera_bundle_tgt->velocity->param_cpy.data(),
            rb->camera_bundle_tgt->bias_gyr->param_cpy.data(),
            rb->camera_bundle_tgt->bias_acc->param_cpy.data(),
            rb->camera_bundle_ref->frame_q->param_cpy.data(),
            rb->camera_bundle_ref->frame_p->param_cpy.data(),
            rb->camera_bundle_ref->velocity->param_cpy.data(),
            rb->camera_bundle_ref->bias_gyr->param_cpy.data(),
            rb->camera_bundle_ref->bias_acc->param_cpy.data()
        };
        rb->param_block_candidate_ptr = {
            rb->camera_bundle_tgt->frame_q->param_new.data(),
            rb->camera_bundle_tgt->frame_p->param_new.data(),
            rb->camera_bundle_tgt->velocity->param_new.data(),
            rb->camera_bundle_tgt->bias_gyr->param_new.data(),
            rb->camera_bundle_tgt->bias_acc->param_new.data(),
            rb->camera_bundle_ref->frame_q->param_new.data(),
            rb->camera_bundle_ref->frame_p->param_new.data(),
            rb->camera_bundle_ref->velocity->param_new.data(),
            rb->camera_bundle_ref->bias_gyr->param_new.data(),
            rb->camera_bundle_ref->bias_acc->param_new.data()
        };

        rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
        rb->PreComputeBlock();

        motion_residual_blocks.push_back(std::move(rb));
        
        cb_tgt->motion_residual_blocks.push_back(motion_residual_blocks.back().get());
        cb_ref->motion_residual_blocks.push_back(motion_residual_blocks.back().get());

        return motion_residual_blocks.back().get();
    }

    ResidualBlockId AddResidualBlock(TinyPreIntegrationPriorFactor* cost_function, LossFunction* loss_function){
        
        auto frame_i = cost_function->piefactor.frame_i;
        auto frame_j = cost_function->piefactor.frame_j;

        ParameterBlockId tgt_frame_q_pb     = AddParameterBlock(frame_i->pose.q.coeffs().data(), 4, PARAM_CAMERA_CONST); // need local parameterization
        ParameterBlockId tgt_frame_p_pb     = AddParameterBlock(frame_i->pose.p.data(), 3, PARAM_CAMERA_CONST);
        ParameterBlockId tgt_frame_v_pb     = AddParameterBlock(frame_i->motion.v.data(), 3, PARAM_MOTION_VELOCITY_CONST);
        ParameterBlockId tgt_frame_bg_pb    = AddParameterBlock(frame_i->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR_CONST);
        ParameterBlockId tgt_frame_ba_pb    = AddParameterBlock(frame_i->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC_CONST);
        ParameterBlockId ref_frame_q_pb     = AddParameterBlock(frame_j->pose.q.coeffs().data(), 4, PARAM_CAMERA);
        ParameterBlockId ref_frame_p_pb     = AddParameterBlock(frame_j->pose.p.data(), 3, PARAM_CAMERA);
        ParameterBlockId ref_frame_v_pb     = AddParameterBlock(frame_j->motion.v.data(), 3, PARAM_MOTION_VELOCITY);
        ParameterBlockId ref_frame_bg_pb    = AddParameterBlock(frame_j->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR);
        ParameterBlockId ref_frame_ba_pb    = AddParameterBlock(frame_j->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC);
        CameraBundleId cb_tgt               = AddCameraBundle(tgt_frame_p_pb, tgt_frame_q_pb, tgt_frame_v_pb, tgt_frame_bg_pb, tgt_frame_ba_pb);
        CameraBundleId cb_ref               = AddCameraBundle(ref_frame_p_pb, ref_frame_q_pb, ref_frame_v_pb, ref_frame_bg_pb, ref_frame_ba_pb);

        std::unique_ptr<MotionResidualBlock> rb = std::make_unique<MotionResidualBlock>(&(cost_function->piefactor), loss_function);
        rb->camera_bundle_tgt = cb_tgt;
        rb->camera_bundle_ref = cb_ref;

        rb->jacobian_ptr = {
            rb->jacobian_tgt_q.data(),
            rb->jacobian_tgt_p.data(),
            rb->jacobian_tgt_v.data(),
            rb->jacobian_tgt_bg.data(),
            rb->jacobian_tgt_ba.data(),
            rb->jacobian_ref_q.data(),
            rb->jacobian_ref_p.data(),
            rb->jacobian_ref_v.data(),
            rb->jacobian_ref_bg.data(),
            rb->jacobian_ref_ba.data()
        };
        rb->param_block_ptr = {
            rb->camera_bundle_tgt->frame_q->param_cpy.data(),
            rb->camera_bundle_tgt->frame_p->param_cpy.data(),
            rb->camera_bundle_tgt->velocity->param_cpy.data(),
            rb->camera_bundle_tgt->bias_gyr->param_cpy.data(),
            rb->camera_bundle_tgt->bias_acc->param_cpy.data(),
            rb->camera_bundle_ref->frame_q->param_cpy.data(),
            rb->camera_bundle_ref->frame_p->param_cpy.data(),
            rb->camera_bundle_ref->velocity->param_cpy.data(),
            rb->camera_bundle_ref->bias_gyr->param_cpy.data(),
            rb->camera_bundle_ref->bias_acc->param_cpy.data()
        };
        rb->param_block_candidate_ptr = {
            rb->camera_bundle_tgt->frame_q->param_new.data(),
            rb->camera_bundle_tgt->frame_p->param_new.data(),
            rb->camera_bundle_tgt->velocity->param_new.data(),
            rb->camera_bundle_tgt->bias_gyr->param_new.data(),
            rb->camera_bundle_tgt->bias_acc->param_new.data(),
            rb->camera_bundle_ref->frame_q->param_new.data(),
            rb->camera_bundle_ref->frame_p->param_new.data(),
            rb->camera_bundle_ref->velocity->param_new.data(),
            rb->camera_bundle_ref->bias_gyr->param_new.data(),
            rb->camera_bundle_ref->bias_acc->param_new.data()
        };

        rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
        rb->PreComputeBlock();

        motion_residual_blocks.push_back(std::move(rb));
        
        cb_tgt->motion_residual_blocks.push_back(motion_residual_blocks.back().get());
        cb_ref->motion_residual_blocks.push_back(motion_residual_blocks.back().get());

        return motion_residual_blocks.back().get();
    }

    ResidualBlockId AddResidualBlock(TinyMarginalizationFactor* cost_function, LossFunction* loss_function){
        
        std::unique_ptr<MarginResidualBlock> rb = std::make_unique<MarginResidualBlock>(cost_function, loss_function);

        const auto frames = cost_function->linearization_frames();
        for(size_t i = 0; i < frames.size(); i++){
            const auto frame = frames[i];
            ParameterBlockId frame_q_pb     = AddParameterBlock(frame->pose.q.coeffs().data(), 4, PARAM_CAMERA);
            ParameterBlockId frame_p_pb     = AddParameterBlock(frame->pose.p.data(), 3, PARAM_CAMERA);
            ParameterBlockId frame_v_pb     = AddParameterBlock(frame->motion.v.data(), 3, PARAM_MOTION_VELOCITY);
            ParameterBlockId frame_bg_pb    = AddParameterBlock(frame->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR);
            ParameterBlockId frame_ba_pb    = AddParameterBlock(frame->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC);
            CameraBundleId cam              = AddCameraBundle(frame_p_pb, frame_q_pb, frame_v_pb, frame_bg_pb, frame_ba_pb);
            rb->camera_bundles.push_back(cam);
            rb->param_block_ptr[i * 5 + 0] = cam->frame_q->param_cpy.data();
            rb->param_block_ptr[i * 5 + 1] = cam->frame_p->param_cpy.data();
            rb->param_block_ptr[i * 5 + 2] = cam->velocity->param_cpy.data();
            rb->param_block_ptr[i * 5 + 3] = cam->bias_gyr->param_cpy.data();
            rb->param_block_ptr[i * 5 + 4] = cam->bias_acc->param_cpy.data();
            rb->param_block_candidate_ptr[i * 5 + 0] = cam->frame_q->param_new.data();
            rb->param_block_candidate_ptr[i * 5 + 1] = cam->frame_p->param_new.data();
            rb->param_block_candidate_ptr[i * 5 + 2] = cam->velocity->param_new.data();
            rb->param_block_candidate_ptr[i * 5 + 3] = cam->bias_gyr->param_new.data();
            rb->param_block_candidate_ptr[i * 5 + 4] = cam->bias_acc->param_new.data();
            rb->jacobians[i * 5 + 0].resize(frames.size() * ES_SIZE, 4);
            rb->jacobians[i * 5 + 1].resize(frames.size() * ES_SIZE, 3);
            rb->jacobians[i * 5 + 2].resize(frames.size() * ES_SIZE, 3);
            rb->jacobians[i * 5 + 3].resize(frames.size() * ES_SIZE, 3);
            rb->jacobians[i * 5 + 4].resize(frames.size() * ES_SIZE, 3);
            rb->jacobians_ptr[i * 5 + 0] = rb->jacobians[i * 5 + 0].data();
            rb->jacobians_ptr[i * 5 + 1] = rb->jacobians[i * 5 + 1].data();
            rb->jacobians_ptr[i * 5 + 2] = rb->jacobians[i * 5 + 2].data();
            rb->jacobians_ptr[i * 5 + 3] = rb->jacobians[i * 5 + 3].data();
            rb->jacobians_ptr[i * 5 + 4] = rb->jacobians[i * 5 + 4].data();
        }

        rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residuals.data(), rb->jacobians_ptr.data());
        rb->PreComputeBlock();

        margin_residual_blocks.push_back(std::move(rb));

        return margin_residual_blocks.back().get();
    }

    CameraBundleId AddCameraBundle(ParameterBlockId p, ParameterBlockId q){
        bool assert1 = pointer_to_camera_bundle_id_map.find(p->param_ptr_raw) == pointer_to_camera_bundle_id_map.end();
        bool assert2 = pointer_to_camera_bundle_id_map.find(q->param_ptr_raw) == pointer_to_camera_bundle_id_map.end();
        if (assert1 && assert2){
            std::unique_ptr<CameraBundle> cb = std::make_unique<CameraBundle>(p, q, p->type);
            camera_bundles.push_back(std::move(cb));
            pointer_to_camera_bundle_id_map[p->param_ptr_raw] = camera_bundles.back().get();
            pointer_to_camera_bundle_id_map[q->param_ptr_raw] = camera_bundles.back().get();
        }else if(assert1 || assert2){
            log_error("one of the camera bundle has been added");
        }else{
            // std::cout << ">>> error: both of the camera bundle has been added" << std::endl;
        }
        return pointer_to_camera_bundle_id_map[p->param_ptr_raw];
    }

    CameraBundleId AddCameraBundle(ParameterBlockId p, ParameterBlockId q, ParameterBlockId v, ParameterBlockId bg, ParameterBlockId ba){
        auto cb = AddCameraBundle(p, q);
        cb->velocity = v;
        cb->bias_gyr = bg;
        cb->bias_acc = ba;
        return cb;
    }

    LandmarkId AddLandmark(ParameterBlockId lm_pb){
        auto it = pointer_to_landmark_id_map.find(lm_pb->param_ptr_raw);
        if(it == pointer_to_landmark_id_map.end()){
            std::unique_ptr<LandmarkBlock> lm = std::make_unique<LandmarkBlock>(lm_pb, lm_pb->type);
            landmarks.push_back(std::move(lm));
            pointer_to_landmark_id_map[lm_pb->param_ptr_raw] = landmarks.back().get();
        }
        return pointer_to_landmark_id_map[lm_pb->param_ptr_raw];
    }

    void PreprocessData(){
        
        if(visual_residual_blocks.size() > 0) _visual_constraint = true;
        if(motion_residual_blocks.size() > 0) _motion_constraint = true;
        if(margin_residual_blocks.size() > 0) _margin_constraint = true;

        num_landmarks                       = landmarks.size();
        num_camera_bundles                  = camera_bundles.size();
        num_visual_residuals                = visual_residual_blocks.size();
        num_motion_residuals                = motion_residual_blocks.size();
        num_margin_residuals                = margin_residual_blocks.size();

        for(size_t i = 0; i < num_camera_bundles; ++i)
            camera_bundles[i]->id = i;
        for(size_t i = 0; i < num_landmarks; ++i)
            landmarks[i]->id = i;
        for(size_t i = 0; i < num_visual_residuals; ++i)
            visual_residual_blocks[i]->id = i;
        for(size_t i = 0; i < num_motion_residuals; ++i)
            motion_residual_blocks[i]->id = i;

        ba_block                            = std::make_unique<BundleAdjustBlocks>();
        ba_block->state_pose                = vector<>::Zero(num_camera_bundles * BA_V_DIM);
        ba_block->state_motion              = vector<>::Zero(num_camera_bundles * BA_M_DIM);
        ba_block->state_landmark            = vector<>::Zero(num_landmarks);

        ba_block->sc_states                 = vector<>::Zero(num_camera_bundles * BA_S_DIM);
        ba_block->sc_matrix                 = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
        ba_block->sc_vector                 = vector<>::Zero(num_camera_bundles * BA_S_DIM);
        ba_block->sc_etfs[0]                = std::vector<matrix<6, 1>>(num_visual_residuals, matrix<6, 1>::Zero());
        ba_block->sc_etfs[1]                = std::vector<matrix<6, 1>>(num_visual_residuals, matrix<6, 1>::Zero());
        ba_block->sc_etes                   = std::vector<matrix1>(num_landmarks, matrix1::Zero());
        ba_block->sc_etes_inv               = std::vector<matrix1>(num_landmarks, matrix1::Zero());
        ba_block->sc_etbs                   = std::vector<vector1>(num_landmarks, vector1::Zero());

        ba_block->visual_matrix             = matrix<>::Zero(num_camera_bundles * BA_V_DIM, num_camera_bundles * BA_V_DIM);
        ba_block->visual_vector             = vector<>::Zero(num_camera_bundles * BA_V_DIM);
        ba_block->motion_matrix             = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
        ba_block->motion_vector             = vector<>::Zero(num_camera_bundles * BA_S_DIM);
        ba_block->margin_matrix             = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
        ba_block->margin_vector             = vector<>::Zero(num_camera_bundles * BA_S_DIM);

#ifdef CFG_OPTIMIZER_DEBUG
        ba_block->visual_jacobian           = matrix<>::Zero(num_visual_residuals * BA_V_RES, num_camera_bundles * BA_V_DIM + num_landmarks);
        ba_block->visual_residual           = vector<>::Zero(num_visual_residuals * BA_V_RES);
        ba_block->visual_hessian            = matrix<>::Zero(num_camera_bundles * BA_V_DIM + num_landmarks, num_camera_bundles * BA_V_DIM + num_landmarks);
        ba_block->visual_ftf                = matrix<>::Zero(num_camera_bundles * BA_V_DIM, num_camera_bundles * BA_V_DIM);
        ba_block->visual_ete                = matrix<>::Zero(num_landmarks, num_landmarks);
        ba_block->visual_etf                = matrix<>::Zero(num_camera_bundles * BA_V_DIM, num_landmarks);

        ba_block->motion_jacobian           = matrix<>::Zero(num_motion_residuals * BA_M_RES, num_camera_bundles * BA_S_DIM);
        ba_block->motion_residual           = vector<>::Zero(num_motion_residuals * BA_M_RES);
        ba_block->motion_hessian            = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);

        ba_block->visual_motion_jacobian    = matrix<>::Zero(num_visual_residuals * BA_V_RES + num_motion_residuals * BA_M_RES, num_camera_bundles * BA_S_DIM + num_landmarks);
        ba_block->visual_motion_residual    = vector<>::Zero(num_visual_residuals * BA_V_RES + num_motion_residuals * BA_M_RES);
        ba_block->visual_motion_hessian     = matrix<>::Zero(num_camera_bundles * BA_S_DIM + num_landmarks, num_camera_bundles * BA_S_DIM + num_landmarks);
        ba_block->visual_motion_ftf         = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
        ba_block->visual_motion_ete         = matrix<>::Zero(num_landmarks, num_landmarks);
        ba_block->visual_motion_etf         = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_landmarks);

        ba_block->margin_jacobian           = matrix<>::Zero(num_camera_bundles * BA_S_RES, num_camera_bundles * BA_S_DIM);
        ba_block->margin_residual           = vector<>::Zero(num_camera_bundles * BA_S_RES);
        ba_block->margin_hessian            = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
#endif

        if(options.verbose){
            log_debug("[TinySolver]: optimizer configuration");
            log_debug("[TinySolver]: prepare memory cost time: %f ms", (GetWallTimeInSeconds() - _solver_start_wall_time) * 1000);
            log_debug("[TinySolver]: camera bundles size: %d", num_camera_bundles);
            log_debug("[TinySolver]: landmarks size: %d", num_landmarks);
            log_debug("[TinySolver]: visual residuals size: %d", num_visual_residuals);
            log_debug("[TinySolver]: motion residuals size: %d", num_motion_residuals);
            log_debug("[TinySolver]: margin residuals size: %d", num_margin_residuals);
        }
    }

    void SetParameterBlockConstant(const double* values, ParamType type){
        if(type == PARAM_CAMERA || type == PARAM_CAMERA_CONST){
            pointer_to_param_id_map[values]->type = PARAM_CAMERA_CONST;
        }else if(type == PARAM_LANDMARK || type == PARAM_LANDMARK_CONST){
            pointer_to_param_id_map[values]->type = PARAM_LANDMARK_CONST;
        }else if(type == PARAM_MOTION_VELOCITY || type == PARAM_MOTION_VELOCITY_CONST){
            pointer_to_param_id_map[values]->type = PARAM_MOTION_VELOCITY_CONST;
        }else if(type == PARAM_MOTION_BIAS_GYR || type == PARAM_MOTION_BIAS_GYR_CONST){
            pointer_to_param_id_map[values]->type = PARAM_MOTION_BIAS_GYR_CONST;
        }else{
            log_error(">>> error: unknown parameter type");
            exit(0);
        }
    }

    void Linearization(){

        if(_reuse_last_state) return;

        for(size_t i = 0; i < visual_residual_blocks.size(); ++i){
            auto& rb = visual_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
            rb->CheckConstantParam();
            rb->PreComputeBlock();
        }

        for(size_t i = 0; i < motion_residual_blocks.size(); ++i){
            auto& rb = motion_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residual.data(), rb->jacobian_ptr.data());
            rb->CheckConstantParam();
            rb->PreComputeBlock();
        }

        for(size_t i = 0; i < margin_residual_blocks.size(); ++i){
            auto& rb = margin_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_ptr.data(), rb->residuals.data(), rb->jacobians_ptr.data());
            rb->CheckConstantParam();
            rb->PreComputeBlock();
        }
    }

    void ComputeMarginBlock(){
        
        if(_reuse_last_state) return;
        if(!_margin_constraint) return;

        auto &margin_matrix = ba_block->margin_matrix;
        auto &margin_vector = ba_block->margin_vector;

        const auto &rb = margin_residual_blocks[0];
        const auto &jacobians_state = rb->jacobians_state;

        for(size_t idx = 0; idx < jacobians_state.size() * jacobians_state.size(); ++idx){
            size_t c = idx / jacobians_state.size(), r = idx % jacobians_state.size();
            const auto &cam_i = rb->camera_bundles[c];
            const auto &cam_j = rb->camera_bundles[r];
            const auto &jac_i = jacobians_state[c];
            const auto &jac_j = jacobians_state[r];
            margin_matrix.block<BA_S_DIM, BA_S_DIM>(cam_i->id * BA_S_DIM, cam_j->id * BA_S_DIM).noalias() += jac_i.transpose() * jac_j;
        }

        for(size_t c = 0; c < jacobians_state.size(); ++c){
            const auto &cam_i = rb->camera_bundles[c];
            const auto &jac_i = jacobians_state[c];
            margin_vector.segment<BA_S_RES>(cam_i->id * BA_S_RES).noalias() += jac_i.transpose() * rb->residuals * -1.0;
        }

#ifdef CFG_OPTIMIZER_DEBUG
        std::ofstream margin_matrix_file("./logs/margin_matrix.txt");
        margin_matrix_file << ba_block->margin_matrix;
        save_matrix_as_image(ba_block->margin_matrix, "./logs/margin_matrix.png");

        std::ofstream margin_vector_file("./logs/margin_vector.txt");
        margin_vector_file << ba_block->margin_vector;
        save_matrix_as_image(ba_block->margin_vector, "./logs/margin_vector.png");

        for(size_t c = 0; c < jacobians_state.size(); ++c){
            for(size_t r = 0; r < jacobians_state.size(); ++r){
                const auto &cam_i = rb->camera_bundles[c];
                const auto &cam_j = rb->camera_bundles[r];
                const auto &jacob = jacobians_state[c];
                ba_block->margin_jacobian.block<BA_S_RES, BA_S_DIM>(cam_j->id * BA_S_RES, cam_i->id * BA_S_DIM).noalias() = jacob.block<BA_S_RES, BA_S_DIM>(cam_j->id * BA_S_RES, 0);
            }
            ba_block->margin_residual.segment<BA_S_RES>(c * BA_S_RES).noalias() = rb->residuals.segment<BA_S_RES>(c * BA_S_RES);
        }

        ba_block->margin_hessian = ba_block->margin_jacobian.transpose() * ba_block->margin_jacobian;

        std::ofstream margin_jacobian_file("./logs/margin_jacobian.txt");
        margin_jacobian_file << ba_block->margin_jacobian;
        save_matrix_as_image(ba_block->margin_jacobian, "./logs/margin_jacobian.png");

        std::ofstream margin_residual_file("./logs/margin_residual.txt");
        margin_residual_file << ba_block->margin_residual;
        save_matrix_as_image(ba_block->margin_residual, "./logs/margin_residual.png");

        std::ofstream margin_hessian_file("./logs/margin_hessian.txt");
        margin_hessian_file << ba_block->margin_hessian;
        save_matrix_as_image(ba_block->margin_hessian, "./logs/margin_hessian.png");

#endif

    }

    void ComputeMotionBlock(){

        if(_reuse_last_state) return;
        if(!_motion_constraint) return;
        
        const size_t frame_num = num_camera_bundles;
        auto &motion_matrix = ba_block->motion_matrix;
        auto &motion_vector = ba_block->motion_vector;

        for(auto &rb: motion_residual_blocks){
            auto &cam1 = rb->camera_bundle_tgt;
            auto &cam2 = rb->camera_bundle_ref;
            
            matrix<BA_S_DIM, BA_S_DIM> dr_dstates_i = matrix<BA_S_DIM, BA_S_DIM>::Zero();
            matrix<BA_S_DIM, BA_S_DIM> dr_dstates_j = matrix<BA_S_DIM, BA_S_DIM>::Zero();
            dr_dstates_i.block<BA_S_RES, BA_V_DIM>(0, ES_Q).noalias() = rb->jacobian_camera[0];
            dr_dstates_i.block<BA_S_RES, BA_M_DIM>(0, ES_V).noalias() = rb->jacobian_motion[0];
            dr_dstates_j.block<BA_S_RES, BA_V_DIM>(0, ES_Q).noalias() = rb->jacobian_camera[1];
            dr_dstates_j.block<BA_S_RES, BA_M_DIM>(0, ES_V).noalias() = rb->jacobian_motion[1];

            motion_matrix.block<BA_S_DIM, BA_S_DIM>(cam1->id * BA_S_DIM, cam1->id * BA_S_DIM).noalias() += dr_dstates_i.transpose() * dr_dstates_i;
            motion_matrix.block<BA_S_DIM, BA_S_DIM>(cam2->id * BA_S_DIM, cam2->id * BA_S_DIM).noalias() += dr_dstates_j.transpose() * dr_dstates_j;
            motion_matrix.block<BA_S_DIM, BA_S_DIM>(cam1->id * BA_S_DIM, cam2->id * BA_S_DIM).noalias() += dr_dstates_i.transpose() * dr_dstates_j;
            motion_matrix.block<BA_S_DIM, BA_S_DIM>(cam2->id * BA_S_DIM, cam1->id * BA_S_DIM).noalias() += dr_dstates_j.transpose() * dr_dstates_i;
            
            motion_vector.segment<BA_S_RES>(cam1->id * BA_S_RES).noalias() += dr_dstates_i.transpose() * rb->residual * -1.0;
            motion_vector.segment<BA_S_RES>(cam2->id * BA_S_RES).noalias() += dr_dstates_j.transpose() * rb->residual * -1.0;
        }

#ifdef CFG_OPTIMIZER_DEBUG
        std::ofstream motion_matrix_file("./logs/motion_matrix.txt");
        motion_matrix_file << ba_block->motion_matrix;
        save_matrix_as_image(ba_block->motion_matrix, "./logs/motion_matrix.png");

        std::ofstream motion_vector_file("./logs/motion_vector.txt");
        motion_vector_file << ba_block->motion_vector;
        save_matrix_as_image(ba_block->motion_vector, "./logs/motion_vector.png");

        for(size_t i = 0; i < num_visual_residuals; ++i){
            const auto &rb = visual_residual_blocks[i];
            const auto &cam1 = rb->camera_bundle_tgt;
            const auto &cam2 = rb->camera_bundle_ref;
            const auto &lm = rb->landmark;

            ba_block->visual_motion_jacobian.block<BA_V_RES, BA_V_DIM>(i * BA_V_RES, cam1->id * BA_S_DIM).noalias() = rb->jacobian_camera[0];
            ba_block->visual_motion_jacobian.block<BA_V_RES, BA_V_DIM>(i * BA_V_RES, cam2->id * BA_S_DIM).noalias() = rb->jacobian_camera[1];
            ba_block->visual_motion_jacobian.block<BA_V_RES, 1>(i * BA_V_RES, frame_num * BA_S_DIM + lm->id) = rb->jacobian_inv_depth;
            ba_block->visual_motion_residual.segment<BA_V_RES>(i * BA_V_RES).noalias() = rb->residual;
        }

        size_t motion_residual_index = num_visual_residuals * BA_V_RES;
        for(size_t i = 0; i < num_motion_residuals; ++i){
            auto &rb = motion_residual_blocks[i];
            auto &cam1 = rb->camera_bundle_tgt;
            auto &cam2 = rb->camera_bundle_ref;

            matrix<BA_S_DIM, BA_S_DIM> dr_dstates_i = matrix<BA_S_DIM, BA_S_DIM>::Zero();
            matrix<BA_S_DIM, BA_S_DIM> dr_dstates_j = matrix<BA_S_DIM, BA_S_DIM>::Zero();
            dr_dstates_i.block<BA_S_RES, BA_V_DIM>(0, ES_Q).noalias() = rb->jacobian_camera[0];
            dr_dstates_i.block<BA_S_RES, BA_M_DIM>(0, ES_V).noalias() = rb->jacobian_motion[0];
            dr_dstates_j.block<BA_S_RES, BA_V_DIM>(0, ES_Q).noalias() = rb->jacobian_camera[1];
            dr_dstates_j.block<BA_S_RES, BA_M_DIM>(0, ES_V).noalias() = rb->jacobian_motion[1];

            ba_block->visual_motion_jacobian.block<BA_M_RES, BA_S_DIM>(motion_residual_index + i * BA_M_RES, cam1->id * BA_S_DIM).noalias() = dr_dstates_i;
            ba_block->visual_motion_jacobian.block<BA_M_RES, BA_S_DIM>(motion_residual_index + i * BA_M_RES, cam2->id * BA_S_DIM).noalias() = dr_dstates_j;
            ba_block->visual_motion_residual.segment<BA_M_RES>(motion_residual_index + i * BA_M_RES).noalias() = rb->residual;
        }

        std::ofstream visual_motion_jacobian_file("./logs/visual_motion_jacobian.txt");
        visual_motion_jacobian_file << ba_block->visual_motion_jacobian;
        save_matrix_as_image(ba_block->visual_motion_jacobian, "./logs/visual_motion_jacobian.png");

        std::ofstream visual_motion_residual_file("./logs/visual_motion_residual.txt");
        visual_motion_residual_file << ba_block->visual_motion_residual;
        save_matrix_as_image(ba_block->visual_motion_residual, "./logs/visual_motion_residual.png");

        ba_block->visual_motion_hessian = ba_block->visual_motion_jacobian.transpose() * ba_block->visual_motion_jacobian;
        std::ofstream visual_motion_hessian_file("./logs/visual_motion_hessian.txt");
        visual_motion_hessian_file << ba_block->visual_motion_hessian;
        save_matrix_as_image(ba_block->visual_motion_hessian, "./logs/visual_motion_hessian.png");

        // ba_block->visual_motion_vector = ba_block->visual_motion_jacobian.transpose() * ba_block->visual_motion_residual;
        // std::ofstream visual_motion_vector_file("./logs/visual_motion_vector.txt");
        // visual_motion_vector_file << ba_block->visual_motion_vector;

        ba_block->visual_motion_ftf = ba_block->visual_motion_hessian.block(0, 0, frame_num * BA_S_DIM, frame_num * BA_S_DIM);
        std::ofstream visual_motion_ftf_file("./logs/visual_motion_ftf.txt");
        visual_motion_ftf_file << ba_block->visual_motion_ftf;
        save_matrix_as_image(ba_block->visual_motion_ftf, "./logs/visual_motion_ftf.png");

        ba_block->visual_motion_ete = ba_block->visual_motion_hessian.block(frame_num * BA_S_DIM, frame_num * BA_S_DIM, num_landmarks, num_landmarks);
        std::ofstream visual_motion_ete_file("./logs/visual_motion_ete.txt");
        visual_motion_ete_file << ba_block->visual_motion_ete;
        save_matrix_as_image(ba_block->visual_motion_ete, "./logs/visual_motion_ete.png");

        ba_block->visual_motion_etf = ba_block->visual_motion_hessian.block(frame_num * BA_S_DIM, 0, num_landmarks, frame_num * BA_S_DIM);
        std::ofstream visual_motion_etf_file("./logs/visual_motion_etf.txt");
        visual_motion_etf_file << ba_block->visual_motion_etf;
        save_matrix_as_image(ba_block->visual_motion_etf, "./logs/visual_motion_etf.png");

        ba_block->visual_motion_vector_sc = ba_block->visual_motion_jacobian.transpose() * ba_block->visual_motion_residual;
        std::ofstream visual_motion_vector_file("./logs/visual_motion_vector_sc.txt");
        visual_motion_vector_file << ba_block->visual_motion_vector_sc;
        save_matrix_as_image(ba_block->visual_motion_vector_sc, "./logs/visual_motion_vector_sc.png");

        ba_block->visual_motion_matrix_sc = ba_block->visual_motion_ftf - ba_block->visual_motion_etf.transpose() * ba_block->visual_motion_ete.inverse() * ba_block->visual_motion_etf;
        std::ofstream visual_motion_matrix_sc_file("./logs/visual_motion_matrix_sc.txt");
        visual_motion_matrix_sc_file << ba_block->visual_motion_matrix_sc;
        save_matrix_as_image(ba_block->visual_motion_matrix_sc, "./logs/visual_motion_matrix_sc.png");
#endif
        
    }

    void ComputeVisualBlock(){

        if(_reuse_last_state) return;
        if(!_visual_constraint) return;

        const size_t frame_num = num_camera_bundles;
        auto &visual_matrix = ba_block->visual_matrix;
        auto &visual_vector = ba_block->visual_vector;

        for(auto& rb: visual_residual_blocks){
            const auto &cam1 = rb->camera_bundle_tgt;
            const auto &cam2 = rb->camera_bundle_ref;

            visual_matrix.block<BA_V_DIM, BA_V_DIM>(cam1->id * BA_V_DIM, cam1->id * BA_V_DIM).noalias() += rb->ftf[0];
            visual_matrix.block<BA_V_DIM, BA_V_DIM>(cam1->id * BA_V_DIM, cam2->id * BA_V_DIM).noalias() += rb->ftf[1];
            visual_matrix.block<BA_V_DIM, BA_V_DIM>(cam2->id * BA_V_DIM, cam1->id * BA_V_DIM).noalias() += rb->ftf[2];
            visual_matrix.block<BA_V_DIM, BA_V_DIM>(cam2->id * BA_V_DIM, cam2->id * BA_V_DIM).noalias() += rb->ftf[3];

            visual_vector.segment<BA_V_DIM>(cam1->id * BA_V_DIM).noalias() += rb->ftb[0];
            visual_vector.segment<BA_V_DIM>(cam2->id * BA_V_DIM).noalias() += rb->ftb[1];
        }

#ifdef CFG_OPTIMIZER_DEBUG
        std::ofstream visual_matrix_file("./logs/visual_matrix.txt");
        visual_matrix_file << ba_block->visual_matrix;
        save_matrix_as_image(ba_block->visual_matrix, "./logs/visual_matrix.png");

        std::ofstream visual_vector_file("./logs/visual_vector.txt");
        visual_vector_file << ba_block->visual_vector;
        save_matrix_as_image(ba_block->visual_vector, "./logs/visual_vector.png");

        for(size_t i = 0; i < num_visual_residuals; ++i){
            const auto& rb = visual_residual_blocks[i];
            const auto& cam1 = rb->camera_bundle_tgt;
            const auto& cam2 = rb->camera_bundle_ref;
            const auto& lm = rb->landmark;
            ba_block->visual_jacobian.block<BA_V_RES, BA_V_DIM>(i * BA_V_RES, cam1->id * BA_V_DIM) = rb->jacobian_camera[0];
            ba_block->visual_jacobian.block<BA_V_RES, BA_V_DIM>(i * BA_V_RES, cam2->id * BA_V_DIM) = rb->jacobian_camera[1];
            ba_block->visual_jacobian.block<BA_V_RES, 1>(i * BA_V_RES, frame_num * BA_V_DIM + lm->id) = rb->jacobian_inv_depth;
            ba_block->visual_residual.segment<BA_V_RES>(i * BA_V_RES) = rb->residual;
        }

        std::ofstream visual_jacobian_file("./logs/visual_jacobian.txt");
        visual_jacobian_file << ba_block->visual_jacobian;
        save_matrix_as_image(ba_block->visual_jacobian, "./logs/visual_jacobian.png");

        std::ofstream visual_residual_file("./logs/visual_residual.txt");
        visual_residual_file << ba_block->visual_residual;
        save_matrix_as_image(ba_block->visual_residual, "./logs/visual_residual.png");

        ba_block->visual_hessian = ba_block->visual_jacobian.transpose() * ba_block->visual_jacobian;
        std::ofstream visual_hessian_file("./logs/visual_hessian.txt");
        visual_hessian_file << ba_block->visual_hessian;
        save_matrix_as_image(ba_block->visual_hessian, "./logs/visual_hessian.png");

        ba_block->visual_ftf = ba_block->visual_hessian.block(0, 0, frame_num * BA_V_DIM, frame_num * BA_V_DIM);
        std::ofstream visual_ftf_file("./logs/visual_ftf.txt");
        save_matrix_as_image(ba_block->visual_ftf, "./logs/visual_ftf.png");

        ba_block->visual_ete = ba_block->visual_hessian.block(frame_num * BA_V_DIM, frame_num * BA_V_DIM, num_landmarks, num_landmarks);
        std::ofstream visual_ete_file("./logs/visual_ete.txt");
        visual_ete_file << ba_block->visual_ete;
        save_matrix_as_image(ba_block->visual_ete, "./logs/visual_ete.png");

        ba_block->visual_etf = ba_block->visual_hessian.block(frame_num * BA_V_DIM, 0, num_landmarks, frame_num * BA_V_DIM);
        std::ofstream visual_etf_file("./logs/visual_etf.txt");
        visual_etf_file << ba_block->visual_etf;
        save_matrix_as_image(ba_block->visual_etf, "./logs/visual_etf.png");

        ba_block->visual_vector_sc = ba_block->visual_jacobian.transpose() * ba_block->visual_residual;
        std::ofstream visual_vector_sc_file("./logs/visual_vector_sc.txt");
        visual_vector_sc_file << ba_block->visual_vector_sc;
        save_matrix_as_image(ba_block->visual_vector_sc, "./logs/visual_vector_sc.png");

        ba_block->visual_matrix_sc = ba_block->visual_ftf - ba_block->visual_etf.transpose() * ba_block->visual_ete.inverse() * ba_block->visual_etf;
        std::ofstream visual_matrix_sc_file("./logs/visual_matrix_sc.txt");
        visual_matrix_sc_file << ba_block->visual_matrix_sc;
        save_matrix_as_image(ba_block->visual_matrix_sc, "./logs/visual_matrix_sc.png");

#endif

    }

    void ComputeSchurComplement(){

        auto &sc_matrix     = ba_block->sc_matrix;
        auto &sc_vector     = ba_block->sc_vector;
        auto &sc_etfs       = ba_block->sc_etfs;
        auto &sc_etes       = ba_block->sc_etes;
        auto &sc_etes_inv   = ba_block->sc_etes_inv;
        auto &sc_etbs       = ba_block->sc_etbs;

        for(size_t idx = 0; idx < num_camera_bundles * num_camera_bundles; ++idx){
            size_t i = idx / num_camera_bundles, j = idx % num_camera_bundles;
            sc_matrix.block<BA_V_DIM, BA_V_DIM>(i * BA_S_DIM, j * BA_S_DIM).noalias() += ba_block->visual_matrix.block<BA_V_DIM, BA_V_DIM>(i * BA_V_DIM, j * BA_V_DIM);
        }

        for(size_t idx = 0; idx < num_camera_bundles; ++idx){
            sc_vector.segment<BA_V_DIM>(idx * BA_S_DIM).noalias() += ba_block->visual_vector.segment<BA_V_DIM>(idx * BA_V_DIM);
        }

        for(size_t i = 0; i < num_visual_residuals; ++i){
            sc_etfs[0][i].noalias() = visual_residual_blocks[i]->etf[0];
            sc_etfs[1][i].noalias() = visual_residual_blocks[i]->etf[1];
        }

        for(size_t i = 0; i < num_landmarks; ++i){
            for (const auto rb : landmarks[i]->residual_blocks) {
                sc_etes[i].noalias() += rb->ete;
                sc_etbs[i].noalias() += rb->etb;
            }
        }

        for(size_t i = 0; i < num_landmarks; ++i){
            sc_etes[i] = sc_etes[i] + matrix1::Identity() / _radius;
            sc_etes_inv[i] = sc_etes[i].inverse();
        }

        size_t r, c;
        for(auto &lm: landmarks){
            const auto& rbs = lm->residual_blocks;
            for(size_t idx = 0; idx < rbs.size() * rbs.size(); ++idx){
                size_t i = idx / rbs.size(), j = idx % rbs.size();
                matrix<6, 1> sc_etfs_etes_inv0 = sc_etfs[0][rbs[i]->id] * sc_etes_inv[lm->id];
                matrix<6, 1> sc_etfs_etes_inv1 = sc_etfs[1][rbs[i]->id] * sc_etes_inv[lm->id];
                r = rbs[i]->camera_bundle_tgt->id * BA_S_DIM, c = rbs[j]->camera_bundle_tgt->id * BA_S_DIM;
                sc_matrix.block<BA_V_DIM, BA_V_DIM>(r, c).noalias() -= sc_etfs_etes_inv0 * sc_etfs[0][rbs[j]->id].transpose();
                r = rbs[i]->camera_bundle_tgt->id * BA_S_DIM, c = rbs[j]->camera_bundle_ref->id * BA_S_DIM;
                sc_matrix.block<BA_V_DIM, BA_V_DIM>(r, c).noalias() -= sc_etfs_etes_inv0 * sc_etfs[1][rbs[j]->id].transpose();
                r = rbs[i]->camera_bundle_ref->id * BA_S_DIM, c = rbs[j]->camera_bundle_tgt->id * BA_S_DIM;
                sc_matrix.block<BA_V_DIM, BA_V_DIM>(r, c).noalias() -= sc_etfs_etes_inv1 * sc_etfs[0][rbs[j]->id].transpose();
                r = rbs[i]->camera_bundle_ref->id * BA_S_DIM, c = rbs[j]->camera_bundle_ref->id * BA_S_DIM;
                sc_matrix.block<BA_V_DIM, BA_V_DIM>(r, c).noalias() -= sc_etfs_etes_inv1 * sc_etfs[1][rbs[j]->id].transpose();
            }

            for(size_t idx = 0; idx < rbs.size(); ++idx){
                matrix1 sc_etes_inv_etbs = sc_etes_inv[lm->id] * sc_etbs[lm->id];
                sc_vector.segment<BA_V_DIM>(rbs[idx]->camera_bundle_tgt->id * BA_S_DIM).noalias() -= sc_etfs[0][rbs[idx]->id] * sc_etes_inv_etbs;
                sc_vector.segment<BA_V_DIM>(rbs[idx]->camera_bundle_ref->id * BA_S_DIM).noalias() -= sc_etfs[1][rbs[idx]->id] * sc_etes_inv_etbs;
            }
        }

        sc_matrix += matrix<>::Identity(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM) / _radius;

#ifdef CFG_OPTIMIZER_DEBUG
        std::ofstream sc_matrix_file("./logs/sc_matrix.txt");
        sc_matrix_file << sc_matrix;
        save_matrix_as_image(sc_matrix, "./logs/sc_matrix.png");

        std::ofstream sc_vector_file("./logs/sc_vector.txt");
        sc_vector_file << sc_vector;
        save_matrix_as_image(sc_vector, "./logs/sc_vector.png");
#endif
    }

    void SchurComplement(){

        ba_block->reset();

        TIME_EVALUATE(ComputeVisualBlock());

        TIME_EVALUATE(ComputeMotionBlock());

        TIME_EVALUATE(ComputeMarginBlock());

        TIME_EVALUATE(ComputeSchurComplement());
    }

    void ComputeDelta(){

        matrix<> real_lhs = matrix<>::Zero(num_camera_bundles * BA_S_DIM, num_camera_bundles * BA_S_DIM);
        vector<> real_rhs = vector<>::Zero(num_camera_bundles * BA_S_DIM);

        real_lhs += ba_block->sc_matrix + ba_block->motion_matrix + ba_block->margin_matrix;
        real_rhs += ba_block->sc_vector + ba_block->motion_vector + ba_block->margin_vector;

#ifdef CFG_OPTIMIZER_DEBUG
        std::ofstream real_lhs_file("./logs/real_lhs.txt");
        real_lhs_file << real_lhs;
        save_matrix_as_image(real_lhs, "./logs/real_lhs.png");

        std::ofstream real_rhs_file("./logs/real_rhs.txt");
        real_rhs_file << real_rhs;
        save_matrix_as_image(real_rhs, "./logs/real_rhs.png");
#endif

        SolveLinearSystemDense(real_lhs, real_rhs, ba_block->sc_states);

        for(size_t i = 0; i < num_camera_bundles; ++i){
            ba_block->state_pose.segment<BA_V_DIM>(i * BA_V_DIM) = ba_block->sc_states.segment<BA_V_DIM>(i * BA_S_DIM);
        }
        for(size_t i = 0; i < num_camera_bundles; ++i){
            ba_block->state_motion.segment<BA_M_DIM>(i * BA_M_DIM) = ba_block->sc_states.segment<BA_M_DIM>(i * BA_S_DIM + BA_V_DIM);
        }
        for(size_t i = 0; i < num_landmarks; ++i){
            const auto &lm = landmarks[i];
            vector1 tmp = vector1::Zero();
            for(const auto& rb: lm->residual_blocks){
                tmp.noalias() += rb->etf[0].transpose() * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_tgt->id * BA_V_DIM);
                tmp.noalias() += rb->etf[1].transpose() * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_ref->id * BA_V_DIM);
            }
            ba_block->state_landmark.segment<1>(i) = ba_block->sc_etes_inv[i] * (ba_block->sc_etbs[i] - tmp);
        }
    }

    void ParamPlusDelta(){

        for(size_t i = 0; i < num_camera_bundles; ++i){
            const auto& cb = camera_bundles[i];
            double* pose = ba_block->state_pose.segment<6>(i * BA_V_DIM).data();
            cb->frame_q->PlusDelta(pose + 0);
            cb->frame_p->PlusDelta(pose + 3);

            if(!_motion_constraint) continue;
            double* motion = ba_block->state_motion.segment<BA_M_DIM>(i * BA_M_DIM).data();
            cb->velocity->PlusDelta(motion + 0);
            cb->bias_gyr->PlusDelta(motion + 3);
            cb->bias_acc->PlusDelta(motion + 6);
        }

        for(size_t i = 0; i < num_landmarks; ++i){
            const auto& lm = landmarks[i];
            lm->landmark->PlusDelta(ba_block->state_landmark.segment<1>(i).data());
        }
    }

    double ComputeModelCostChange(){

        double delta = 0.0;
        for(size_t i = 0; i < num_visual_residuals; ++i){
            const auto& rb = visual_residual_blocks[i];
            vector<2> model_residual = vector<2>::Zero();
            model_residual += rb->jacobian_camera[0] * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_tgt->id * BA_V_DIM);
            model_residual += rb->jacobian_camera[1] * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_ref->id * BA_V_DIM);
            model_residual += rb->jacobian_inv_depth * ba_block->state_landmark.segment<1>(rb->landmark->id * 1);
            vector<2> tmp = model_residual / 2.0 + rb->residual;
            delta -= model_residual.dot(tmp);
        }

        for(size_t i = 0; i < num_motion_residuals; ++i){
            const auto& rb = motion_residual_blocks[i];
            vector<15> model_residual = vector<15>::Zero();
            model_residual += rb->jacobian_camera[0] * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_tgt->id * BA_V_DIM);
            model_residual += rb->jacobian_camera[1] * ba_block->state_pose.segment<BA_V_DIM>(rb->camera_bundle_ref->id * BA_V_DIM);
            model_residual += rb->jacobian_motion[0] * ba_block->state_motion.segment<BA_M_DIM>(rb->camera_bundle_tgt->id * BA_M_DIM);
            model_residual += rb->jacobian_motion[1] * ba_block->state_motion.segment<BA_M_DIM>(rb->camera_bundle_ref->id * BA_M_DIM);
            vector<15> tmp = model_residual / 2.0 + rb->residual;
            delta -= model_residual.dot(tmp);
        }

        for(size_t i = 0; i < num_margin_residuals; ++i){
            const auto& rb = margin_residual_blocks[i];
            const auto& cams = rb->camera_bundles;
            vector<> states = vector<>::Zero(cams.size() * 15);
            for(size_t c = 0; c < cams.size(); ++c){
                const auto& cam = cams[c];
                states.segment<BA_V_DIM>(c * BA_S_DIM) = ba_block->state_pose.segment<BA_V_DIM>(cam->id * BA_V_DIM);
                states.segment<BA_M_DIM>(c * BA_S_DIM + BA_V_DIM) = ba_block->state_motion.segment<BA_M_DIM>(cam->id * BA_M_DIM);
            }
            vector<> model_residual = rb->jacobians_matrix * states;
            vector<> tmp = model_residual / 2.0 + rb->residuals;
            delta -= model_residual.dot(tmp);
        }
        return delta;
    }

    double ComputeCurrentEvalutePointNorm(){
        double state_norm = 0.0;
        for(size_t i = 0; i < num_camera_bundles; ++i){

            if(!_visual_constraint) continue;
            state_norm += camera_bundles[i]->frame_q->XSquareNorm();
            state_norm += camera_bundles[i]->frame_p->XSquareNorm();

            if(!_motion_constraint) continue;
            state_norm += camera_bundles[i]->velocity->XSquareNorm();
            state_norm += camera_bundles[i]->bias_gyr->XSquareNorm();
            state_norm += camera_bundles[i]->bias_acc->XSquareNorm();
        }
        for(size_t i = 0; i< num_landmarks; ++i){
            state_norm += landmarks[i]->landmark->XSquareNorm();
        }
        return std::sqrt(state_norm);
    }

    double ComputeResiduals(){
        double cost = 0.0;
        for(size_t i = 0; i < visual_residual_blocks.size(); ++i){
            cost += visual_residual_blocks[i]->ComputeCost();
        }
        for(size_t i = 0; i < motion_residual_blocks.size(); ++i){
            cost += motion_residual_blocks[i]->ComputeCost();
        }
        for(size_t i = 0; i < margin_residual_blocks.size(); ++i){
            cost += margin_residual_blocks[i]->ComputeCost();
        }
        return cost / 2.0;
    }

    double ComputeGradient(){
        double grad_norm = 0.0;
        for(const auto& rb: visual_residual_blocks){
            grad_norm += (rb->jacobian_camera[0].transpose() * rb->residual).squaredNorm();
            grad_norm += (rb->jacobian_camera[1].transpose() * rb->residual).squaredNorm();
            grad_norm += (rb->jacobian_inv_depth.transpose() * rb->residual).squaredNorm();
        }
        for(const auto& rb: motion_residual_blocks){
            grad_norm += (rb->jacobian_camera[0].transpose() * rb->residual).squaredNorm();
            grad_norm += (rb->jacobian_camera[1].transpose() * rb->residual).squaredNorm();
            grad_norm += (rb->jacobian_motion[0].transpose() * rb->residual).squaredNorm();
            grad_norm += (rb->jacobian_motion[1].transpose() * rb->residual).squaredNorm();
        }
        for(const auto& rb: margin_residual_blocks){
            grad_norm += (rb->jacobians_matrix.transpose() * rb->residuals).squaredNorm();
        }
        return std::sqrt(2 * grad_norm);
    }

    double ComputeStepNorm(){
        double step_square = 0.0;
        for(const auto& cb: camera_bundles){

            if(!_visual_constraint) continue;
            step_square += cb->frame_q->StepSquareNorm();
            step_square += cb->frame_p->StepSquareNorm();

            if(!_motion_constraint) continue;
            step_square += cb->velocity->StepSquareNorm();
            step_square += cb->bias_gyr->StepSquareNorm();
            step_square += cb->bias_acc->StepSquareNorm();
        }
        for(const auto& lm: landmarks){
            step_square += lm->landmark->StepSquareNorm();
        }
        return sqrt(step_square);
    }

    double ComputeCandidateResiduals(){
        double cost = 0.0;
        for(size_t i = 0; i < visual_residual_blocks.size(); ++i){
            auto &rb = visual_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_candidate_ptr.data(), rb->residual_candidate.data(), nullptr);
            cost += rb->ComputeCost(rb->residual_candidate);
        }
        for(size_t i = 0; i < motion_residual_blocks.size(); ++i){
            auto &rb = motion_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_candidate_ptr.data(), rb->residual_candidate.data(), nullptr);
            cost += rb->ComputeCost(rb->residual_candidate);
        }
        for(size_t i = 0; i < margin_residual_blocks.size(); ++i){
            auto &rb = margin_residual_blocks[i];
            rb->factor->Evaluate(rb->param_block_candidate_ptr.data(), rb->residuals_candidate.data(), nullptr);
            cost += rb->ComputeCost(rb->residuals_candidate);
        }
        return cost / 2.0;
    }

    void UpdateParameterBlock(){

        for(size_t i = 0; i < num_camera_bundles; ++i){
            const auto& cb = camera_bundles[i];

            if(!_visual_constraint) continue;
            cb->frame_p->Update();
            cb->frame_q->Update();
        
            if(!_motion_constraint) continue;
            cb->velocity->Update();
            cb->bias_gyr->Update();
            cb->bias_acc->Update();
        }

        for(size_t i = 0; i < num_landmarks; ++i){
            const auto& lm = landmarks[i];
            lm->landmark->Update();
        }
    }

    void StepRejected(double step_quality){
        // TODO: this may be used in LM
        _radius = _radius / _decrease_factor;
        _radius = std::max(options.min_radius, std::min(options.max_radius, _radius));
        _decrease_factor *= 2.0;

        _reuse_last_state = true;
    }

    void StepAccepted(double step_quality){
        // TODO: this may be used in LM
        _radius = _radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * step_quality - 1.0, 3));
        _radius = std::max(options.min_radius, std::min(options.max_radius, _radius));
        _decrease_factor = 2.0;

        _is_succeed = true;
        _reuse_last_state = false;
    }

    bool CheckTerminationCriteriaAndLog(){
        
        StepInfo(iteration_summary.iteration, true);

        iteration_summary.iteration++;
        iteration_summary.iteration_start_wall_time = GetWallTimeInSeconds();

        if(MaxSolverIterationsReached()) 
            return false;

        if(MaxSolverTimeReached()) 
            return false;

        if(_num_invalid_steps >= options.max_num_invalid_steps) 
            return false;

        return true;
    }

    bool IterationZero(){

        iteration_summary.cost                  = ComputeResiduals();
        iteration_summary.cost_change           = 0.0;
        iteration_summary.step_is_valid         = false;
        iteration_summary.tr_radius             = options.radius;

        return true;
    }

    bool UpdateIterationSummary(){

        iteration_summary.cost                  = ComputeResiduals();
        iteration_summary.cost_change           = iteration_summary.cost - iteration_summary.candidate_cost;
        iteration_summary.model_cost_change     = ComputeModelCostChange();
        iteration_summary.relative_decrease     = iteration_summary.cost_change / iteration_summary.model_cost_change;
        iteration_summary.step_norm             = ComputeStepNorm();
        iteration_summary.tr_radius             = _radius;

        return iteration_summary.model_cost_change > 0.0 && iteration_summary.relative_decrease > options.min_relative_decrease;
    }

    bool ComputeCandidatePointAndEvaluateCost(){
        
        TIME_EVALUATE(Linearization());

        TIME_EVALUATE(SchurComplement());

        TIME_EVALUATE(ComputeDelta());

        TIME_EVALUATE(ParamPlusDelta());
        
        TIME_EVALUATE(UpdateIterationSummary());
        
        return iteration_summary.model_cost_change > 0.0 && iteration_summary.relative_decrease > options.min_relative_decrease;
    }

    bool HandSuccessfulStep(){
        
        UpdateParameterBlock();

        StepAccepted(iteration_summary.relative_decrease);

        iteration_summary.cost = iteration_summary.candidate_cost;
        iteration_summary.step_is_valid = true;

        _num_invalid_steps = 0;

        return true;
    }

    bool HandInvalidStep(){
        
        StepRejected(0);

        iteration_summary.step_is_valid = false;

        _num_invalid_steps++;
        
        return true;
    }

    void SolveLinearSystemDense(const matrix<> &lhs, const vector<> &rhs, vector<> &dx){

        Eigen::HouseholderQR<Eigen::MatrixXd> qr = lhs.householderQr();
        dx = qr.solve(rhs);
    }

    void AcceptOptimizationResult(){
        for(auto& cb: camera_bundles){

            if(!_visual_constraint) continue;
            cb->frame_q->AcceptResult();
            cb->frame_p->AcceptResult();

            if(!_motion_constraint) continue;
            cb->velocity->AcceptResult();
            cb->bias_gyr->AcceptResult();
            cb->bias_acc->AcceptResult();
        }

        for(auto& lm: landmarks){
            lm->landmark->AcceptResult();
        }
    }

    void StepInfo(size_t iter, bool verbose){

        if(!verbose) return;

        double current_time = GetWallTimeInSeconds();

        if(iter == 0){
            std::ostringstream oss;
            oss << std::setw(4) << "iter"
                << std::setw(10) << "cost"
                << std::setw(17) << "cost_change"
                // << std::setw(12) << "|gradient|"
                << std::setw(10) << "|step|"
                << std::setw(13) << "tr_ratio"
                << std::setw(13) << "tr_radius"
                << std::setw(12) << "accept"
                << std::setw(12) << "iter_time"
                << std::setw(12) << "total_time";

            if(options.verbose)
                log_debug("[TinySolver]: %s", oss.str().c_str());

        }
        std::ostringstream oss;
        oss << std::setw(4) << iter
            << std::setw(14) << std::scientific << std::setprecision(6) << iteration_summary.cost
            << std::setw(12) << std::scientific << std::setprecision(2) << iteration_summary.cost_change
            // << std::setw(12) << std::scientific << std::setprecision(2) << iteration_summary.gradient_norm
            << std::setw(12) << std::scientific << std::setprecision(2) << iteration_summary.step_norm
            << std::setw(12) << std::scientific << std::setprecision(2) << iteration_summary.relative_decrease
            << std::setw(12) << std::scientific << std::setprecision(2) << iteration_summary.tr_radius
            << std::setw(12) << std::noboolalpha << iteration_summary.step_is_valid
            << std::setw(12) << std::scientific << std::setprecision(2) << current_time - iteration_summary.iteration_start_wall_time
            << std::setw(12) << std::scientific << std::setprecision(2) << current_time - _solver_start_wall_time;

        if(options.verbose)
            log_debug("[TinySolver]: %s", oss.str().c_str());
    }

    bool ParameterToleranceReached(){
        double x_norm = ComputeCurrentEvalutePointNorm();
        double step_size_tolerance = options.parameter_tolerance * (x_norm + options.parameter_tolerance);
        if (iteration_summary.step_norm > step_size_tolerance) {
            return false;
        }
        // std::cout << "ParameterToleranceReached" << std::endl;
        return true;
    }

    bool FunctionToleranceReached(){
        double absolute_function_tolerance = options.function_tolerance * iteration_summary.cost;
        if (std::fabs(iteration_summary.cost_change) > absolute_function_tolerance) {
            return false;
        }
        // std::cout << "FunctionToleranceReached" << std::endl;
        return true;
    }

    bool MaxSolverIterationsReached() {
        if (iteration_summary.iteration < options.max_num_iterations) {
            return false;
        }
        // std::cout << "MaxSolverIterationsReached" << std::endl;
        return true;
    }

    bool MaxSolverTimeReached(){
        double total_solver_time = GetWallTimeInSeconds() - _solver_start_wall_time;
        if (total_solver_time < options.max_solver_time_in_seconds) {
            return false;
        }
        // std::cout << "MaxSolverTimeReached" << std::endl;
        return false;
    }

    bool GradientToleranceReached(){
        /*
        TODO: Not implemented yet!
        */
        return false;
    }

#ifdef CFG_OPTIMIZER_DEBUG
    cv::Mat save_matrix_as_image(const matrix<>& mat, const std::string& name){
        cv::Mat img(mat.rows(), mat.cols(), CV_8UC3, cv::Scalar(255, 255, 255));
        for(int i = 0; i < mat.rows(); ++i){
            for(int j = 0; j < mat.cols(); ++j){
                if(fabs(mat(i, j)) > 1.0e-6)
                    img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 204, 0);
            }
        }
        cv::imwrite(name, img);
        return img;
    }
#endif

};

}

#endif