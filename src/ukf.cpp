#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  // initial state vector
  x_ = VectorXd(5);
  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // start
  is_initialized_ = false;
  // State dimension, which is the size of State Vector x_: [pos1,pos2,vel_abs,yaw_angle,yaw_rate]
  n_x_ = 5;
  // Augmented state dimension, which add process noise vector to the above State Vector, noise vector is [std_a_, std_yawdd_] 
  n_aug_ = 7;
  // Sigma point spreading parameter, for ukf with noise
  lambda_ = 3 - n_aug_;
  // Number of sigma points 
  n_sig_ = 2 * n_aug_ + 1;
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  // Weights vector 
  weights_ = VectorXd(n_sig_);
  weights_(0) = (double) lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }
  // end

}

UKF::~UKF() {
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // start
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    
    P_.topLeftCorner(5,5).setIdentity();
    P_ = P_ * 2.0;
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float t_rho = meas_package.raw_measurements_[0];
      float t_phi = meas_package.raw_measurements_[1];
      float t_rhodot = meas_package.raw_measurements_[2];

      float t_px = t_rho * cos(t_phi);
      float t_py = t_rho * sin(t_phi);
      float t_vx = t_rhodot * cos(t_phi);
      float t_vy = t_rhodot * sin(t_phi);
      float v    = sqrt(t_vx * t_vx + t_vy * t_vy);
      
      x_ << t_px, t_py, v, 0.0, 0.0;
    }else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;

      if (fabs(x_(0)) < 0.01 and fabs(x_(1)) < 0.01) {
        x_(0) = 0.01;
        x_(1) = 0.01;
      }
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    // exit out of routine as intialised
    return;
  }
  
  // Calculating the time delta
  double d_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  
  // Perform prediction
  Prediction(d_t);
  
  // Perform update 
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
  // end
}

void UKF::Prediction(double delta_t) {

  // start
  VectorXd x_aug = VectorXd(7); 
  x_aug.head(5) = x_;
  // Init noise values 
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  // Noise Matrix t_Q
  MatrixXd t_Q = MatrixXd(2, 2);
  t_Q << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;
  
  // Augmented State Matrix
  MatrixXd t_P_aug = MatrixXd(7, 7);

  // Augmented Sigma Point Matrix
  MatrixXd t_Xsig_aug = MatrixXd(n_aug_, n_sig_); 

  t_P_aug.fill(0.0);

  t_P_aug.topLeftCorner(5, 5) = P_;
  t_P_aug.bottomRightCorner(2, 2) = t_Q;
  
  MatrixXd t_L = t_P_aug.llt().matrixL();
  
  t_Xsig_aug.col(0) = x_aug;
  
  for (int i = 0; i < n_aug_; i++) {
    t_Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * t_L.col(i);
    t_Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * t_L.col(i);
  }
  
  /*
  Perform predict sigma points matrix
  */
  
  for (int i = 0; i < n_sig_; i++) {
    double t_p_x      = t_Xsig_aug(0, i);
    double t_p_y      = t_Xsig_aug(1, i);
    double t_v        = t_Xsig_aug(2, i);
    double t_yaw      = t_Xsig_aug(3, i);
    double t_yawd     = t_Xsig_aug(4, i);
    double t_nu_a     = t_Xsig_aug(5, i);
    double t_nu_yawdd = t_Xsig_aug(6, i);
    
    double t_px_p, t_py_p;
    
    if (fabs(t_yawd) > 0.001) {
      t_px_p = t_p_x + t_v/t_yawd * (sin(t_yaw + t_yawd * delta_t) - sin(t_yaw));
      t_py_p = t_p_y + t_v/t_yawd * (cos(t_yaw) - cos(t_yaw + t_yawd * delta_t));
    } else {
      t_px_p = t_p_x + t_v * delta_t * cos(t_yaw);
      t_py_p = t_p_y + t_v * delta_t * sin(t_yaw);
    }

    double t_v_p = t_v;
    double t_yaw_p = t_yaw + t_yawd * delta_t;
    double t_yawd_p = t_yawd;
    
    // Adding noise
    t_px_p = t_px_p + 0.5 * t_nu_a * delta_t * delta_t * cos(t_yaw);
    t_py_p = t_py_p + 0.5 * t_nu_a * delta_t * delta_t * sin(t_yaw);
    t_v_p  = t_v_p + t_nu_a * delta_t;
    
    t_yaw_p = t_yaw_p + 0.5 * t_nu_yawdd * delta_t * delta_t;
    t_yawd_p = t_yawd_p + t_nu_yawdd * delta_t;
    
    Xsig_pred_(0, i) = t_px_p;
    Xsig_pred_(1, i) = t_py_p;
    Xsig_pred_(2, i) = t_v_p;
    Xsig_pred_(3, i) = t_yaw_p;
    Xsig_pred_(4, i) = t_yawd_p;
  }
  
  /*
  Perform predict mean and covariance
  */
  
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++){

    VectorXd t_x_diff = Xsig_pred_.col(i) - x_;
    while (t_x_diff(3) > M_PI)  t_x_diff(3) -= 2. * M_PI;
    while (t_x_diff(3) < -1 * M_PI) t_x_diff(3) += 2. * M_PI;
    
    P_ = P_ + weights_(i) * t_x_diff * t_x_diff.transpose();
  }
  // end
}



void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // start
  int n_z = 2;
  
  MatrixXd t_Zsig = Xsig_pred_.topLeftCorner(n_z, n_sig_);
  VectorXd t_z_pred = VectorXd(n_z);
  MatrixXd t_S = MatrixXd(n_z,n_z);
  
  t_z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
    t_z_pred = t_z_pred + weights_(i) *  t_Zsig.col(i);
  }

  t_S.fill(0.0);
  for (int i=0; i < n_sig_; i++){
    VectorXd z_diff = t_Zsig.col(i) - t_z_pred;
    
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2. * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2. * M_PI;
    }
    
    t_S = t_S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd t_R = MatrixXd(2,2);
  t_R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;
  t_S = t_S + t_R;
  
  MatrixXd t_Tc = MatrixXd(n_x_, n_z);

  // Cross correlation matrix
  t_Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    
    VectorXd t_z_diff = t_Zsig.col(i) - t_z_pred;

    while (t_z_diff(1) > M_PI) {
      t_z_diff(1) -= 2. * M_PI;
    }
    while (t_z_diff(1) < -M_PI) {
      t_z_diff(1) += 2. * M_PI;
    }
    
    VectorXd t_x_diff = Xsig_pred_.col(i) - x_;
    while (t_x_diff(3) > M_PI) {
      t_x_diff(3) -= 2. * M_PI;
    }
    while (t_x_diff(3) < -M_PI) {
      t_x_diff(3) += 2. * M_PI;
    }
    
    t_Tc = t_Tc + weights_(i) * t_x_diff * t_z_diff.transpose();
  }

  VectorXd t_z = meas_package.raw_measurements_;
  MatrixXd t_K = t_Tc * t_S.inverse();
  VectorXd t_z_diff = t_z - t_z_pred;
  
  while (t_z_diff(1) > M_PI) {
    t_z_diff(1) -= 2. * M_PI;
  }
  while (t_z_diff(1) < -M_PI) {
    t_z_diff(1) += 2. * M_PI;
  }
  
  // Update mean matrix and covariance matrix state
  x_ = x_ + t_K * t_z_diff;
  P_ = P_ - t_K * t_S * t_K.transpose();
  
  NIS_laser_ = t_z_diff.transpose() * t_S.inverse() * t_z_diff;
  // end
}



void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // start
  int n_z = 3;
  
  MatrixXd t_Zsig = MatrixXd(n_z, n_sig_);
  
  for (int i = 0; i < n_sig_; i++) {
    double t_p_x = Xsig_pred_(0, i);
    double t_p_y = Xsig_pred_(1, i);
    double t_v   = Xsig_pred_(2, i);
    double t_yaw = Xsig_pred_(3, i);
      
    double t_v1 = cos(t_yaw) * t_v;
    double t_v2 = sin(t_yaw) * t_v;

    t_Zsig(0, i) = sqrt(t_p_x * t_p_x + t_p_y * t_p_y);             // rho
    t_Zsig(1, i) = atan2(t_p_y, t_p_x);                             // phi
    t_Zsig(2, i) = (t_p_x * t_v1 + t_p_y * t_v2 ) / t_Zsig(0, i);   // rhodot
  }

  VectorXd t_z_pred = VectorXd(n_z);
    
  t_z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
    t_z_pred = t_z_pred + weights_(i) *  t_Zsig.col(i);
  }

  MatrixXd t_S = MatrixXd(n_z, n_z);

  t_S.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
    VectorXd t_z_diff = t_Zsig.col(i) - t_z_pred;
    while (t_z_diff(1) > M_PI) {
      t_z_diff(1) -= 2. * M_PI;
    }
    while (t_z_diff(1) < -M_PI) {
      t_z_diff(1) += 2. * M_PI;
    }
        
    t_S = t_S + weights_(i) * t_z_diff * t_z_diff.transpose();
  }
    
  MatrixXd t_R = MatrixXd(3, 3);
  t_R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_ * std_radrd_;
  
  t_S = t_S + t_R;
  
  /*
  Perform Radar State Update
  */
  
  MatrixXd t_Tc = MatrixXd(n_x_, n_z);
  t_Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    VectorXd t_z_diff = t_Zsig.col(i) - t_z_pred;

    while (t_z_diff(1) > M_PI) {
      t_z_diff(1) -= 2. * M_PI;
    }
    while (t_z_diff(1) < -M_PI) {
      t_z_diff(1) += 2. * M_PI;
    }
    
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2. * M_PI;
    }
    
    t_Tc = t_Tc + weights_(i) * x_diff * t_z_diff.transpose();
  }
  
  VectorXd t_z = meas_package.raw_measurements_;
  MatrixXd t_K = t_Tc * t_S.inverse();
  VectorXd t_z_diff = t_z - t_z_pred;
  while (t_z_diff(1) > M_PI) {
    t_z_diff(1) -= 2. * M_PI;
  }
  while (t_z_diff(1) < -M_PI) {
    t_z_diff(1) += 2. * M_PI;
  }
  
  // Update mean matrix and covariance matrix
  x_ = x_ + t_K * t_z_diff;
  P_ = P_ - t_K * t_S * t_K.transpose();
  
  NIS_radar_ = t_z_diff.transpose() * t_S.inverse() * t_z_diff;

  // end
}
