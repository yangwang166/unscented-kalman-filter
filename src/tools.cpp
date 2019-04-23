#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  // start

  VectorXd t_rmse(4);
  t_rmse << 0, 0, 0, 0;
  
  if(estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground truth data" << endl;
    return t_rmse;
  }
  
  for(unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd r= estimations[i] - ground_truth[i];
    r = r.array() * r.array();
    t_rmse += r;
  }
  
  t_rmse = t_rmse / estimations.size();
  t_rmse = t_rmse.array().sqrt();
  return t_rmse; 

  // end
}
