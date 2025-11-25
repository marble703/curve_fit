#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

// Simple residual for (x,z) given timestamp
struct TrajResidual {
    TrajResidual(double x_obs, double z_obs, double t, double x0, double z0)
        : x_obs_(x_obs), z_obs_(z_obs), time_(t), x0_(x0), z0_(z0) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        const T vx0 = params[0];
        const T vz0 = params[1];
        const T logk = params[2];
        const T k = ceres::exp(logk);
        const T t = T(time_);
        const T exp_term = ceres::exp(-k * t);
        const T one_minus = T(1.0) - exp_term;
        const T invk = T(1.0) / k;
        const T g = T(9.81);

        T x_model = T(x0_) + vx0 * invk * one_minus;
        T z_model = T(z0_) + vz0 * invk * one_minus - (g / k) * t + (g / (k * k)) * one_minus;

        residuals[0] = T(x_obs_) - x_model;
        residuals[1] = T(z_obs_) - z_model;
        return true;
    }

private:
    const double x_obs_, z_obs_, time_, x0_, z0_;
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

    // Optional undistort: convert to normalized image points then to pixels
    std::vector<cv::Point2d> undistorted = pixels;
    if (!dist.empty()) {
        // undistortPoints yields normalized coords; convert back to pixel coords
        std::vector<cv::Point2d> norm_pts;
        cv::undistortPoints(pixels, norm_pts, K, dist);
        for (int i = 0; i < N; ++i) {
            cv::Point2d p = norm_pts[i];
            // back to pixel coordinates: [u v 1]^T = K * [x y 1]^T
            cv::Mat pt = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
            cv::Mat pix = K * pt;
            undistorted[i].x = pix.at<double>(0,0) / pix.at<double>(2,0);
            undistorted[i].y = pix.at<double>(1,0) / pix.at<double>(2,0);
        }
    }

    // Backproject pixels to plane Y=0 using ray-plane intersection
    // Camera model: P_cam = R * P_world + t  => P_world = R^T (P_cam - t) = R^T P_cam - R^T t
    // Camera center in world: C_world = -R^T t
    // Ray direction in camera: d_cam = K^{-1} [u,v,1]^T
    // Ray direction in world : d_world = R^T d_cam
    // Ray: P(s) = C_world + s * d_world; set Y=0 => s = -C_world.y / d_world.y
    cv::Mat K_inv = K.inv();
    cv::Mat R_t = R.t();
    cv::Mat C_world = -R_t * t; // 3x1

    std::vector<double> xs(N), zs(N);
    for (int i = 0; i < N; ++i) {
        cv::Mat uv = (cv::Mat_<double>(3,1) << undistorted[i].x, undistorted[i].y, 1.0);
        cv::Mat d_cam = K_inv * uv;
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

    double duration = times.back() - times.front();
    if (duration <= 0.0) duration = 1.0;
    double vx_guess = (xs.back() - xs.front()) / duration;
    double vz_guess = (zs.back() - zs.front()) / duration;
    if (!std::isfinite(vx_guess)) vx_guess = 5.0;
    if (!std::isfinite(vz_guess)) vz_guess = 5.0;
    double logk_guess = std::log(0.5);

    double params[3] = {vx_guess, vz_guess, logk_guess};

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TrajResidual, 2, 3>(
            new TrajResidual(xs[i], zs[i], times[i], x0, z0));
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost, loss, params);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    double k = std::exp(params[2]);
    double vx0 = params[0];
    double vz0 = params[1];
    double sigma = 9.81 / (k * k);
    std::cout << "Estimated parameters:\n";
    std::cout << " vx0  = " << vx0 << " m/s\n";
    std::cout << " vz0  = " << vz0 << " m/s\n";
    std::cout << " k    = " << k << " (b/m)\n";
    std::cout << " sigma= " << sigma << "\n";

    // Output fitted trajectory (sampled)
    std::ofstream ofs("fitted_traj.txt");
    const int samples = 50;
    double t_min = times.front();
    double t_max = times.back();
    for (int i = 0; i < samples; ++i) {
        double t = t_min;
        if (samples > 1) t = t_min + (t_max - t_min) * double(i) / double(samples - 1);
        double exp_term = std::exp(-k * t);
        double one_minus = 1.0 - exp_term;
        double x = x0 + vx0 / k * one_minus;
        double z = z0 + vz0 / k * one_minus - (9.81 / k) * t + (9.81 / (k * k)) * one_minus;
        ofs << x << " " << z << "\n";
    }
    ofs.close();

    std::cout << "Fitted trajectory written to 'fitted_traj.txt' (x z per line)." << std::endl;
    return 0;
}
