#include <slamtool/estimation/tinysolver/tiny_reprojection_factor.h>
#include <slamtool/estimation/tinysolver/tiny_preintegration_factor.h>
#include <slamtool/estimation/tinysolver/tiny_marginalization_factor.h>
#include <slamtool/estimation/tinysolver/linear_problem.h>
#include <slamtool/estimation/tinysolver/loss_function.h>

#include <slamtool/estimation/quaternion_parameterization.h>
#include <slamtool/estimation/solver.h>
#include <slamtool/estimation/state.h>
#include <slamtool/map/frame.h>
#include <slamtool/estimation/solver.h>

namespace slamtool {

struct TinySolverDetails: public Solver::SolverDetails {

    std::unique_ptr<LinearProblem>                  problem;
    std::unique_ptr<QuaternionParameterization>     quaternion_parameterization;
    std::unique_ptr<LossFunction>                   cauchy_loss;
    std::unique_ptr<LossFunction>                   trivial_loss;
    
    void create_problem() override {
        problem                                     = std::make_unique<LinearProblem>();
        cauchy_loss                                 = std::make_unique<CauchyLoss>(1.0);
        trivial_loss                                = std::make_unique<TrivialLoss>();
        quaternion_parameterization                 = std::make_unique<QuaternionParameterization>();
    }

    bool solve(bool verbose) override{
        LinearProblem::Options solver_options;
        solver_options.verbose                      = verbose;
        solver_options.max_num_iterations           = config()->solver_iteration_limit();
        solver_options.max_solver_time_in_seconds   = config()->solver_time_limit();
        solver_options.num_threads                  = 1;
        return problem->Solve(solver_options);
    }

    void add_frame_states(Frame *frame, bool with_motion = true){
        problem->AddParameterBlock(frame->pose.q.coeffs().data(), 4, PARAM_CAMERA, quaternion_parameterization.get());
        problem->AddParameterBlock(frame->pose.p.data(), 3, PARAM_CAMERA);
        if (frame->tag(FT_FIX_POSE)) {
            problem->SetParameterBlockConstant(frame->pose.q.coeffs().data(), PARAM_CAMERA_CONST);
            problem->SetParameterBlockConstant(frame->pose.p.data(), PARAM_CAMERA_CONST);
        }
        if (with_motion) {
            problem->AddParameterBlock(frame->motion.v.data(), 3, PARAM_MOTION_VELOCITY);
            problem->AddParameterBlock(frame->motion.bg.data(), 3, PARAM_MOTION_BIAS_GYR);
            problem->AddParameterBlock(frame->motion.ba.data(), 3, PARAM_MOTION_BIAS_ACC);
            if (frame->tag(FT_FIX_MOTION)) {
                problem->SetParameterBlockConstant(frame->motion.v.data(), PARAM_MOTION_VELOCITY_CONST);
                problem->SetParameterBlockConstant(frame->motion.bg.data(), PARAM_MOTION_BIAS_GYR_CONST);
                problem->SetParameterBlockConstant(frame->motion.ba.data(), PARAM_MOTION_BIAS_ACC_CONST);
            }
        }
    }

    void add_track_states(Track *track, bool fix_track){
        problem->AddParameterBlock(&(track->landmark.inv_depth), 1, PARAM_LANDMARK);
        if(fix_track) problem->SetParameterBlockConstant(&(track->landmark.inv_depth), PARAM_LANDMARK_CONST);
    }

    void add_factor(ReprojectionErrorFactor *rpefactor) override {
        TinyReprojectionErrorFactor *rpecost = static_cast<TinyReprojectionErrorFactor *>(rpefactor);
        problem->AddResidualBlock(rpecost, cauchy_loss.get());
    }

    void add_factor(ReprojectionPriorFactor *rppfactor) override {
        TinyReprojectionPriorFactor *rppcost = static_cast<TinyReprojectionPriorFactor *>(rppfactor);
        problem->AddResidualBlock(rppcost, cauchy_loss.get());
    }

    void add_factor(RotationPriorFactor *ropfactor) override {
        log_error("Not implemented RotationPriorFactor yet");
    }

    void add_factor(PreIntegrationErrorFactor *piefactor) override {
        TinyPreIntegrationErrorFactor *piecost = static_cast<TinyPreIntegrationErrorFactor *>(piefactor);
        problem->AddResidualBlock(piecost, trivial_loss.get());
    }

    void add_factor(PreIntegrationPriorFactor *pipfactor) override {
        TinyPreIntegrationPriorFactor *pipcost = static_cast<TinyPreIntegrationPriorFactor *>(pipfactor);
        problem->AddResidualBlock(pipcost, trivial_loss.get());
    }

    void add_factor(MarginalizationFactor *factor) override {
        TinyMarginalizationFactor *marginalization_factor = static_cast<TinyMarginalizationFactor *>(factor);
        problem->AddResidualBlock(marginalization_factor, trivial_loss.get());
    }
};

}