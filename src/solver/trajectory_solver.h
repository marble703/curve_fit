#ifndef CURVE_FIT_TRAJECTORY_SOLVER_H_
#define CURVE_FIT_TRAJECTORY_SOLVER_H_

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

namespace curve_fit {

struct TauFitResult {
    double sigma_x = 0.0;
    double sigma_z = 0.0;
    double delta_tau = 0.0;
    double sigma_g = 0.0;

    double k = 0.0;
    double vx0 = 0.0;
    double vz0 = 0.0;
};

struct TrajectorySolverOptions {
    int max_num_iterations = 200;
    double huber_threshold = 1.0;
    bool print_progress = true;
};

struct TrajectorySolveResult {
    TauFitResult fit;
    ceres::Solver::Summary summary;
};

struct CurveFitConfig {
    std::string camera_path;
    std::string pixels_path;
    bool override_x0 = false;
    double x0 = 0.0;
    bool override_z0 = false;
    double z0 = 0.0;
    std::string fitted_trajectory_path = "fitted_traj.txt";
};

struct CurveFitResult {
    TrajectorySolveResult solver_result;
    std::vector<double> xs;
    std::vector<double> zs;
    double x0 = 0.0;
    double z0 = 0.0;
    std::string fitted_trajectory_path;
};

class TrajectorySolver {
public:
    explicit TrajectorySolver(const TrajectorySolverOptions& options = TrajectorySolverOptions())
        : options_(options) {}

    TrajectorySolveResult Solve(const std::vector<double>& xs,
                                const std::vector<double>& zs,
                                double x0,
                                double z0) const {
        if (xs.empty() || xs.size() != zs.size()) {
            throw std::invalid_argument("TrajectorySolver expects xs and zs of equal non-zero size");
        }

        const int N = static_cast<int>(xs.size());
        double sigma_x_guess = xs.back() - xs.front();
        double sigma_z_guess = zs.back() - zs.front();
        if (std::abs(sigma_x_guess) < 1e-6) sigma_x_guess = 1.0;
        if (std::abs(sigma_z_guess) < 1e-6) sigma_z_guess = 1.0;

        double log_delta_tau_guess = std::log(3.0 / std::max(1, N - 1));
        double log_sigma_g_guess = std::log(1.0);

        double params[4] = {sigma_x_guess, sigma_z_guess, log_delta_tau_guess, log_sigma_g_guess};

        ceres::Problem problem;
        for (int i = 0; i < N; ++i) {
            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TrajResidual, 2, 4>(
                new TrajResidual(xs[i], zs[i], i, x0, z0));
            ceres::LossFunction* loss = new ceres::HuberLoss(options_.huber_threshold);
            problem.AddResidualBlock(cost, loss, params);
        }

        ceres::Solver::Options ceres_options;
        ceres_options.minimizer_progress_to_stdout = options_.print_progress;
        ceres_options.max_num_iterations = options_.max_num_iterations;

        TrajectorySolveResult result;
        ceres::Solve(ceres_options, &problem, &result.summary);

        const double sigma_x = params[0];
        const double sigma_z = params[1];
        const double delta_tau = std::exp(params[2]);
        const double sigma_g = std::exp(params[3]);

        result.fit.sigma_x = sigma_x;
        result.fit.sigma_z = sigma_z;
        result.fit.delta_tau = delta_tau;
        result.fit.sigma_g = sigma_g;

        const double g = 9.81;
        result.fit.k = std::sqrt(g / sigma_g);
        result.fit.vx0 = sigma_x * result.fit.k;
        result.fit.vz0 = sigma_z * result.fit.k;

        return result;
    }

private:
    struct TrajResidual {
        TrajResidual(double x_obs, double z_obs, int index, double x0, double z0)
            : x_obs_(x_obs), z_obs_(z_obs), index_(index), x0_(x0), z0_(z0) {}

        template <typename T>
        bool operator()(const T* const params, T* residuals) const {
            const T sigma_x = params[0];
            const T sigma_z = params[1];
            const T delta_tau = ceres::exp(params[2]);
            const T sigma_g = ceres::exp(params[3]);

            const T tau = T(index_) * delta_tau;
            const T exp_term = ceres::exp(-tau);
            const T one_minus = T(1.0) - exp_term;

            T x_model = T(x0_) + sigma_x * one_minus;
            T z_model = T(z0_) + sigma_z * one_minus - sigma_g * tau + sigma_g * one_minus;

            residuals[0] = T(x_obs_) - x_model;
            residuals[1] = T(z_obs_) - z_model;
            return true;
        }

    private:
        const double x_obs_, z_obs_;
        const int index_;
        const double x0_, z0_;
    };

    TrajectorySolverOptions options_;
};

namespace internal {

inline bool ParseArray(const std::string& key,
                       int expect,
                       const std::string& content,
                       std::vector<double>& out) {
    out.clear();
    std::istringstream all(content);
    std::string line;
    size_t start = std::string::npos;
    size_t cur = 0;
    while (std::getline(all, line)) {
        std::string trimmed = line;
        const size_t first = trimmed.find_first_not_of(" \t");
        trimmed = (first == std::string::npos) ? "" : trimmed.substr(first);
        if (trimmed.rfind(key, 0) == 0) {
            start = cur;
            break;
        }
        cur += line.size() + 1;
    }
    if (start == std::string::npos) return false;
    size_t dp = content.find("data:", start);
    if (dp == std::string::npos) return false;
    size_t lb = content.find('[', dp);
    size_t rb = content.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string inner = content.substr(lb + 1, rb - lb - 1);
    for (char& c : inner) if (c == ',') c = ' ';
    std::istringstream iss(inner);
    double v;
    while (iss >> v) {
        out.push_back(v);
    }
    return expect <= 0 || static_cast<int>(out.size()) == expect;
}

inline bool FallbackLoadCamera(const std::string& path,
                               cv::Mat& K,
                               cv::Mat& dist,
                               cv::Mat& R,
                               cv::Mat& t) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<double> kv, dv, rv, tv;
    if (!ParseArray("K:", 9, content, kv) ||
        !ParseArray("R:", 9, content, rv) ||
        !ParseArray("t:", 3, content, tv)) {
        return false;
    }
    ParseArray("dist:", -1, content, dv);
    K = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) K.at<double>(i / 3, i % 3) = kv[i];
    R = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) R.at<double>(i / 3, i % 3) = rv[i];
    t = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) t.at<double>(i, 0) = tv[i];
    dist = cv::Mat(1, static_cast<int>(dv.size()), CV_64F);
    for (size_t i = 0; i < dv.size(); ++i) dist.at<double>(0, static_cast<int>(i)) = dv[i];
    std::cerr << "Fallback parsing successful." << std::endl;
    return true;
}

inline bool LoadCamera(const std::string& path,
                       cv::Mat& K,
                       cv::Mat& dist,
                       cv::Mat& R,
                       cv::Mat& t) {
    cv::FileStorage fs;
    try {
        fs.open(path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["K"] >> K;
            fs["dist"] >> dist;
            fs["R"] >> R;
            fs["t"] >> t;
            fs.release();
            return true;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV FileStorage parse failed: " << e.what() << std::endl;
    }
    std::cerr << "Falling back to simple YAML text parser." << std::endl;
    return FallbackLoadCamera(path, K, dist, R, t);
}

inline std::vector<cv::Point2d> NormalizePixels(const std::vector<cv::Point2d>& pixels,
                                                 const cv::Mat& K,
                                                 const cv::Mat& dist) {
    const int N = static_cast<int>(pixels.size());
    std::vector<cv::Point2d> normalized(N);
    if (!dist.empty()) {
        std::vector<cv::Point2d> pts;
        cv::undistortPoints(pixels, pts, K, dist);
        for (int i = 0; i < N; ++i) {
            normalized[i] = pts[i];
        }
        return normalized;
    }
    cv::Mat K_inv = K.inv();
    for (int i = 0; i < N; ++i) {
        cv::Mat uv = (cv::Mat_<double>(3, 1) << pixels[i].x, pixels[i].y, 1.0);
        cv::Mat nc = K_inv * uv;
        normalized[i].x = nc.at<double>(0, 0) / nc.at<double>(2, 0);
        normalized[i].y = nc.at<double>(1, 0) / nc.at<double>(2, 0);
    }
    return normalized;
}

inline void BackprojectToPlane(const std::vector<cv::Point2d>& norm_pts,
                               const cv::Mat& R,
                               const cv::Mat& t,
                               std::vector<double>& xs,
                               std::vector<double>& zs) {
    const int N = static_cast<int>(norm_pts.size());
    xs.resize(N);
    zs.resize(N);
    cv::Mat R_t = R.t();
    cv::Mat C_world = -R_t * t;
    for (int i = 0; i < N; ++i) {
        cv::Mat d_cam = (cv::Mat_<double>(3, 1) << norm_pts[i].x, norm_pts[i].y, 1.0);
        cv::Mat d_world = R_t * d_cam;
        double dy = d_world.at<double>(1, 0);
        double Cy = C_world.at<double>(1, 0);
        if (std::abs(dy) < 1e-9) {
            xs[i] = d_world.at<double>(0, 0) * 1e6;
            zs[i] = d_world.at<double>(2, 0) * 1e6;
        } else {
            double s = -Cy / dy;
            xs[i] = C_world.at<double>(0, 0) + s * d_world.at<double>(0, 0);
            zs[i] = C_world.at<double>(2, 0) + s * d_world.at<double>(2, 0);
        }
    }
}

inline void WriteFittedTrajectory(const std::string& path,
                                  const TauFitResult& fit,
                                  double x0,
                                  double z0,
                                  int samples,
                                  int n_points) {
    std::ofstream ofs(path);
    const double tau_max = (n_points > 1) ? (n_points - 1) * fit.delta_tau : fit.delta_tau;
    for (int i = 0; i < samples; ++i) {
        double tau = 0.0;
        if (samples > 1) tau = tau_max * double(i) / double(samples - 1);
        double exp_term = std::exp(-tau);
        double one_minus = 1.0 - exp_term;
        double x = x0 + fit.sigma_x * one_minus;
        double z = z0 + fit.sigma_z * one_minus - fit.sigma_g * tau + fit.sigma_g * one_minus;
        ofs << x << " " << z << "\n";
    }
}

} // namespace internal

inline CurveFitResult RunCurveFit(const CurveFitConfig& config,
                                  const TrajectorySolverOptions& solver_options) {
    if (config.camera_path.empty() || config.pixels_path.empty()) {
        throw std::invalid_argument("Camera and pixel paths must be provided");
    }

    cv::Mat K, dist, R, t;
    if (!internal::LoadCamera(config.camera_path, K, dist, R, t)) {
        throw std::runtime_error("Failed to load camera configuration from " + config.camera_path);
    }

    std::vector<cv::Point2d> pixels;
    std::ifstream ifs(config.pixels_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open pixel file " + config.pixels_path);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double u, v;
        if (!(iss >> u >> v)) continue;
        pixels.emplace_back(u, v);
    }

    if (pixels.empty()) {
        throw std::runtime_error("No pixel observations loaded from " + config.pixels_path);
    }

    const int N = static_cast<int>(pixels.size());
    if (N < 6) {
        std::cerr << "Warning: fewer than 6 points may be unstable." << std::endl;
    }

    std::vector<cv::Point2d> normalized = internal::NormalizePixels(pixels, K, dist);
    std::vector<double> xs, zs;
    internal::BackprojectToPlane(normalized, R, t, xs, zs);

    std::cerr << "First 5 backprojected (x,z):";
    for (int i = 0; i < std::min(5, N); ++i) {
        std::cerr << " (" << xs[i] << "," << zs[i] << ")";
    }
    std::cerr << std::endl;

    double x0 = xs[0];
    double z0 = zs[0];
    if (config.override_x0) x0 = config.x0;
    if (config.override_z0) z0 = config.z0;

    TrajectorySolver solver(solver_options);
    curve_fit::TrajectorySolveResult solve_result = solver.Solve(xs, zs, x0, z0);

    internal::WriteFittedTrajectory(config.fitted_trajectory_path,
                                    solve_result.fit,
                                    x0,
                                    z0,
                                    50,
                                    N);

    CurveFitResult result;
    result.solver_result = std::move(solve_result);
    result.xs = std::move(xs);
    result.zs = std::move(zs);
    result.x0 = x0;
    result.z0 = z0;
    result.fitted_trajectory_path = config.fitted_trajectory_path;
    return result;
}

} // namespace curve_fit
#endif  // CURVE_FIT_TRAJECTORY_SOLVER_H_
