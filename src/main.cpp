#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

// τ-parameterized residual: no dependence on absolute time
// Model: x(τ) = x0 + sigma_x * (1 - exp(-τ))
//        z(τ) = z0 + sigma_z * (1 - exp(-τ)) - sigma_g * τ + sigma_g * (1 - exp(-τ))
// where sigma_g = g / k^2 is derived from sigma_x, sigma_z and trajectory geometry
// τ_i = i * delta_tau for equally spaced observations
struct TrajResidual {
    TrajResidual(double x_obs, double z_obs, int index, double x0, double z0)
        : x_obs_(x_obs), z_obs_(z_obs), index_(index), x0_(x0), z0_(z0) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        // params: [sigma_x, sigma_z, log_delta_tau, log_sigma_g]
        const T sigma_x = params[0];
        const T sigma_z = params[1];
        const T log_delta_tau = params[2];
        const T log_sigma_g = params[3];
        const T delta_tau = ceres::exp(log_delta_tau);
        const T sigma_g = ceres::exp(log_sigma_g);
        
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

// Read UV points from a text file: each line "u v"
bool LoadPixels(const std::string& path, std::vector<cv::Point2d>& pixels) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    pixels.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double u, v;
        if (!(iss >> u >> v)) continue;
        pixels.emplace_back(u, v);
    }
    return true;
}

bool LoadTimes(const std::string& path, std::vector<double>& times) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    times.clear();
    double value = 0;
    while (ifs >> value) {
        times.push_back(value);
    }
    return !times.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " camera.yml pixels.txt [x0 z0] [mode]\n";
        std::cout << "  camera.yml: OpenCV FileStorage with K, dist (optional), R, t\n";
        std::cout << "  pixels.txt: lines of 'u v' (image pixels)\n";
        std::cout << "  [x0 z0]: optional start point on plane Y=0 (defaults to first backprojected point)\n";
        std::cout << "  [mode]: 'fixed_tau' (default) or 'opt_tau' (not implemented)\n";
        return 1;
    }

    const std::string cam_path = argv[1];
    const std::string pix_path = argv[2];

    std::string times_path;
    bool use_times_file = false;
    double user_x0 = 0.0;
    double user_z0 = 0.0;
    bool has_user_x0 = false;
    bool has_user_z0 = false;
    int arg_idx = 3;
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "--times" && arg_idx + 1 < argc) {
            times_path = argv[++arg_idx];
            use_times_file = true;
        } else if (arg == "--x0" && arg_idx + 1 < argc) {
            user_x0 = std::stod(argv[++arg_idx]);
            has_user_x0 = true;
        } else if (arg == "--z0" && arg_idx + 1 < argc) {
            user_z0 = std::stod(argv[++arg_idx]);
            has_user_z0 = true;
        } else {
            break;
        }
        ++arg_idx;
    }
    if (!use_times_file && arg_idx < argc) {
        std::ifstream test(argv[arg_idx]);
        if (test.good()) {
            times_path = argv[arg_idx];
            use_times_file = true;
            ++arg_idx;
        }
    }
    if (!has_user_x0 && !has_user_z0 && arg_idx + 1 < argc) {
        user_x0 = std::stod(argv[arg_idx]);
        user_z0 = std::stod(argv[arg_idx + 1]);
        has_user_x0 = has_user_z0 = true;
    }

    cv::Mat K, dist, R, t;
    bool fs_ok = false;
    cv::FileStorage fs;
    try {
        fs.open(cam_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["K"] >> K;
            fs["dist"] >> dist;
            fs["R"] >> R;
            fs["t"] >> t;
            fs.release();
            fs_ok = true;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV FileStorage parse failed: " << e.what() << std::endl;
    }

    if (!fs_ok) {
        std::cerr << "Falling back to simple YAML text parser." << std::endl;
        std::ifstream ifs(cam_path);
        if (!ifs.is_open()) { std::cerr << "Cannot open " << cam_path << std::endl; return 1; }
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        auto parse_array = [&](const std::string& key, int expect, std::vector<double>& out) -> bool {
            out.clear();
            std::istringstream all(content); std::string line; size_t start = std::string::npos, cur = 0;
            while (std::getline(all, line)) {
                std::string tr = line; size_t p = tr.find_first_not_of(" \t"); if (p != std::string::npos) tr = tr.substr(p); else tr = "";
                if (tr.rfind(key, 0) == 0) { start = cur; break; } cur += line.size() + 1;
            }
            if (start == std::string::npos) return false;
            size_t dp = content.find("data:", start); if (dp == std::string::npos) return false;
            size_t lb = content.find('[', dp), rb = content.find(']', lb); if (lb == std::string::npos || rb == std::string::npos) return false;
            std::string inner = content.substr(lb+1, rb-lb-1); for (char &c : inner) if (c == ',') c = ' ';
            std::istringstream iss(inner); double v; while (iss >> v) out.push_back(v);
            return expect <= 0 || (int)out.size() == expect;
        };
        std::vector<double> kv, dv, rv, tv;
        if (!parse_array("K:", 9, kv) || !parse_array("R:", 9, rv) || !parse_array("t:", 3, tv)) {
            std::cerr << "Fallback parser failed." << std::endl; return 1;
        }
        parse_array("dist:", -1, dv);
        K = cv::Mat(3,3,CV_64F); for (int i = 0; i < 9; ++i) K.at<double>(i/3, i%3) = kv[i];
        R = cv::Mat(3,3,CV_64F); for (int i = 0; i < 9; ++i) R.at<double>(i/3, i%3) = rv[i];
        t = cv::Mat(3,1,CV_64F); for (int i = 0; i < 3; ++i) t.at<double>(i,0) = tv[i];
        dist = cv::Mat(1, (int)dv.size(), CV_64F); for (size_t i = 0; i < dv.size(); ++i) dist.at<double>(0,(int)i) = dv[i];
        std::cerr << "Fallback parsing successful." << std::endl;
    }

    if (K.empty() || R.empty() || t.empty()) {
        std::cerr << "camera.yml must contain K, R, t (dist optional)" << std::endl;
        return 1;
    }

    std::vector<cv::Point2d> pixels;
    if (!LoadPixels(pix_path, pixels)) {
        std::cerr << "Failed to load pixels from: " << pix_path << std::endl;
        return 1;
    }
    const int N = static_cast<int>(pixels.size());
    if (N < 6) std::cerr << "Warning: fewer than 6 points may be unstable." << std::endl;

    std::vector<double> times;
    if (use_times_file) {
        if (!LoadTimes(times_path, times)) {
            std::cerr << "Failed to load times from: " << times_path << std::endl;
            return 1;
        }
        std::cerr << "Using timestamps from " << times_path << " (" << times.size() << " entries)." << std::endl;
    } else {
        double t_max = 1.0;
        times.resize(N);
        for (int i = 0; i < N; ++i) times[i] = t_max * double(i) / double(std::max(1, N - 1));
        std::cerr << "No times file provided; using normalized [0,1] spacing." << std::endl;
    }
    if (static_cast<int>(times.size()) != N) {
        std::cerr << "Times count (" << times.size() << ") does not match pixel count (" << N << ")." << std::endl;
        return 1;
    }

    // Undistort and get normalized camera coordinates
    // undistortPoints outputs normalized coords directly (no need to multiply by K then K^{-1})
    std::vector<cv::Point2d> norm_pts(N);
    if (!dist.empty()) {
        cv::undistortPoints(pixels, norm_pts, K, dist);
    } else {
        // No distortion: manually convert pixels to normalized coords
        cv::Mat K_inv_tmp = K.inv();
        for (int i = 0; i < N; ++i) {
            cv::Mat uv = (cv::Mat_<double>(3,1) << pixels[i].x, pixels[i].y, 1.0);
            cv::Mat nc = K_inv_tmp * uv;
            norm_pts[i].x = nc.at<double>(0,0) / nc.at<double>(2,0);
            norm_pts[i].y = nc.at<double>(1,0) / nc.at<double>(2,0);
        }
    }

    // Backproject normalized coords to plane Y=0 using ray-plane intersection
    // Ray direction in camera (normalized coords): d_cam = [nx, ny, 1]^T
    // Ray direction in world: d_world = R^T * d_cam
    // Camera center in world: C_world = -R^T * t
    // Ray: P(s) = C_world + s * d_world; set Y=0 => s = -C_world.y / d_world.y
    cv::Mat R_t = R.t();
    cv::Mat C_world = -R_t * t; // 3x1

    std::vector<double> xs(N), zs(N);
    for (int i = 0; i < N; ++i) {
        cv::Mat d_cam = (cv::Mat_<double>(3,1) << norm_pts[i].x, norm_pts[i].y, 1.0);
        cv::Mat d_world = R_t * d_cam;
        double dy = d_world.at<double>(1,0);
        double Cy = C_world.at<double>(1,0);
        if (std::abs(dy) < 1e-9) {
            // ray parallel to Y=0 plane, use projection as fallback (point at infinity)
            xs[i] = d_world.at<double>(0,0) * 1e6;
            zs[i] = d_world.at<double>(2,0) * 1e6;
        } else {
            double s = -Cy / dy;
            xs[i] = C_world.at<double>(0,0) + s * d_world.at<double>(0,0);
            zs[i] = C_world.at<double>(2,0) + s * d_world.at<double>(2,0);
        }
    }

    // Debug print first few backprojected points
    std::cerr << "First 5 backprojected (x,z):";
    for (int i = 0; i < std::min(5, N); ++i) std::cerr << " (" << xs[i] << "," << zs[i] << ")";
    std::cerr << std::endl;

    double x0 = xs[0];
    double z0 = zs[0];
    if (has_user_x0) x0 = user_x0;
    if (has_user_z0) z0 = user_z0;

    // τ-parameterization: estimate sigma_x, sigma_z from trajectory span
    // sigma_x ≈ x_final - x0, sigma_z ≈ z_max - z0 (rough estimate)
    double sigma_x_guess = xs.back() - xs.front();
    double sigma_z_guess = zs.back() - zs.front();
    if (std::abs(sigma_x_guess) < 1e-6) sigma_x_guess = 1.0;
    if (std::abs(sigma_z_guess) < 1e-6) sigma_z_guess = 1.0;
    
    // Initial guess for delta_tau: assume trajectory spans τ ∈ [0, ~3] (reasonable decay)
    double log_delta_tau_guess = std::log(3.0 / std::max(1, N - 1));
    // Initial guess for sigma_g = g/k^2: rough estimate from trajectory curvature
    double log_sigma_g_guess = std::log(1.0);  // Will be refined by optimizer

    double params[4] = {sigma_x_guess, sigma_z_guess, log_delta_tau_guess, log_sigma_g_guess};

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TrajResidual, 2, 4>(
            new TrajResidual(xs[i], zs[i], i, x0, z0));
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost, loss, params);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    
    // Extract τ-parameterized results
    double sigma_x = params[0];
    double sigma_z = params[1];
    double delta_tau = std::exp(params[2]);
    double sigma_g = std::exp(params[3]);
    
    // Recover physical parameters (if time scale is known)
    // sigma_g = g / k^2  =>  k = sqrt(g / sigma_g)
    // sigma_x = vx0 / k  =>  vx0 = sigma_x * k
    // sigma_z = vz0 / k  =>  vz0 = sigma_z * k
    double g = 9.81;
    double k_recovered = std::sqrt(g / sigma_g);
    double vx0_recovered = sigma_x * k_recovered;
    double vz0_recovered = sigma_z * k_recovered;
    
    std::cout << "\nEstimated τ-parameterized results (time-independent):\n";
    std::cout << " sigma_x   = " << sigma_x << " m (characteristic x-displacement)\n";
    std::cout << " sigma_z   = " << sigma_z << " m (characteristic z-displacement)\n";
    std::cout << " delta_tau = " << delta_tau << " (dimensionless time step)\n";
    std::cout << " sigma_g   = " << sigma_g << " m (gravity length scale = g/k^2)\n";
    std::cout << "\nRecovered physical parameters (assuming g=9.81 m/s^2):\n";
    std::cout << " k    = " << k_recovered << " 1/s (drag coefficient)\n";
    std::cout << " vx0  = " << vx0_recovered << " m/s\n";
    std::cout << " vz0  = " << vz0_recovered << " m/s\n";

    // Output fitted trajectory (sampled using τ parameterization)
    std::ofstream ofs("fitted_traj.txt");
    const int samples = 50;
    double tau_max = (N - 1) * delta_tau;  // Total τ span
    for (int i = 0; i < samples; ++i) {
        double tau = 0.0;
        if (samples > 1) tau = tau_max * double(i) / double(samples - 1);
        double exp_term = std::exp(-tau);
        double one_minus = 1.0 - exp_term;
        double x = x0 + sigma_x * one_minus;
        double z = z0 + sigma_z * one_minus - sigma_g * tau + sigma_g * one_minus;
        ofs << x << " " << z << "\n";
    }
    ofs.close();

    std::cout << "Fitted trajectory written to 'fitted_traj.txt' (x z per line)." << std::endl;
    return 0;
}
