#include <iostream>
#include <string>

#include "solver/trajectory_solver.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " camera.yml pixels.txt [--x0 val] [--z0 val] [x0 z0]" << std::endl;
        std::cerr << "  camera.yml: OpenCV FileStorage with K, dist (optional), R, t" << std::endl;
        std::cerr << "  pixels.txt: lines of \"u v\" (image pixels)." << std::endl;
        std::cerr << "  Optional arguments: provide --x0 and --z0 to override the plane intersection guess." << std::endl;
        std::cerr << "  The solver now uses τ parameterization, so timestamps are optional." << std::endl;
        return 1;
    }

    curve_fit::CurveFitConfig config;
    config.camera_path = argv[1];
    config.pixels_path = argv[2];

    int arg_idx = 3;
    while (arg_idx < argc) {
        const std::string arg = argv[arg_idx];
        if (arg == "--x0" && arg_idx + 1 < argc) {
            config.override_x0 = true;
            config.x0 = std::stod(argv[++arg_idx]);
        } else if (arg == "--z0" && arg_idx + 1 < argc) {
            config.override_z0 = true;
            config.z0 = std::stod(argv[++arg_idx]);
        } else {
            break;
        }
        ++arg_idx;
    }

    if (!config.override_x0 && !config.override_z0 && arg_idx + 1 < argc) {
        config.override_x0 = config.override_z0 = true;
        config.x0 = std::stod(argv[arg_idx]);
        config.z0 = std::stod(argv[arg_idx + 1]);
    }

    try {
        curve_fit::TrajectorySolverOptions options;
        curve_fit::CurveFitResult result = curve_fit::RunCurveFit(config, options);

        std::cout << result.solver_result.summary.FullReport() << std::endl;
        const curve_fit::TauFitResult& fit = result.solver_result.fit;
        std::cout << "\nEstimated τ-parameterized results (time-independent):" << std::endl;
        std::cout << " sigma_x   = " << fit.sigma_x << " m (characteristic x-displacement)" << std::endl;
        std::cout << " sigma_z   = " << fit.sigma_z << " m (characteristic z-displacement)" << std::endl;
        std::cout << " delta_tau = " << fit.delta_tau << " (dimensionless time step)" << std::endl;
        std::cout << " sigma_g   = " << fit.sigma_g << " m (gravity length scale = g/k^2)" << std::endl;
        std::cout << "\nRecovered physical parameters (assuming g=9.81 m/s^2):" << std::endl;
        std::cout << " k    = " << fit.k << " 1/s (drag coefficient)" << std::endl;
        std::cout << " vx0  = " << fit.vx0 << " m/s" << std::endl;
        std::cout << " vz0  = " << fit.vz0 << " m/s" << std::endl;
        std::cout << "Fitted trajectory written to '" << result.fitted_trajectory_path << "' (x z per line)." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
