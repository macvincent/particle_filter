/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */
#include "../../eigen/Eigen/Dense"
#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  std::default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  

  for (int i = 0; i < num_particles; i++) {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta= dist_theta(gen);
    Particle sample(i, sample_x, sample_y, sample_theta, 1);
    particles.push_back(sample);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (auto& particle : particles){
    double theta = particle.theta;
    particle.x += (velocity/yaw_rate)*(sin(theta + yaw_rate*delta_t) - sin(theta));
    particle.y += (velocity/yaw_rate)*(-cos(theta + yaw_rate*delta_t) + cos(theta));
    particle.theta += yaw_rate*delta_t;

    std::default_random_engine gen;
    // This line creates a normal (Gaussian) distribution for each of the predictions
    normal_distribution<double> dist_x(particle.x, std_pos[0]);
    normal_distribution<double> dist_y(particle.y, std_pos[1]);
    normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predictions, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the observations measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for(auto& obs : observations){
      float distance = dist(obs.x, obs.y, predictions[0].x, predictions[0].y);
      float result = 0;
      for(int i = 1; i < predictions.size(); i++){
        int temp_distance = dist(obs.x, obs.y, predictions[i].x, predictions[i].y);
          if(temp_distance < distance){
              distance = temp_distance;
              result = i;
          }
      }
      obs.id = predictions[result].id;
      obs.gausian = find_gaussian(obs.x, obs.y, predictions[result].x, predictions[result].y);
    }
}
LandmarkObs transformation(float x_particle, float y_particle, float heading_particle, float x_obs, float y_obs){
    float teta = heading_particle*3.142/180; // coonvert heading to radian
    VectorXd obs = VectorXd(3);
    MatrixXd tf = MatrixXd(3,3);
    obs << x_obs, y_obs, 1;
    tf <<  cos(teta), -sin(teta), x_particle,
            sin(teta), cos(teta), y_particle,
            0, 0, 1;
    VectorXd result = VectorXd(3);
    result =  tf * obs;
    LandmarkObs return_value;
    return_value.x = result(0);
    return_value.y = result(1);
    return return_value;

}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                    vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  std::default_random_engine gen;

  for (auto particle: particles){
    vector<LandmarkObs> transformed_observation;
    for(auto obs: observations){
      transformed_observation.push_back(transformation(particle.x, particle.y, particle.theta, obs.x, obs.y));
    }
    vector<LandmarkObs> possible_landmarks;
    for(auto landmark: map_landmarks.landmark_list){
      if(abs(landmark.x_f - particle.x) <= sensor_range && abs(landmark.y_f - particle.y) <= sensor_range){
        LandmarkObs temp;
        temp.x = landmark.x_f;
        temp.y = landmark.y_f;
        temp.id = landmark.id_i;
        possible_landmarks.push_back(temp);
      }
    }
    dataAssociation(possible_landmarks, transformed_observation);
    for(auto obs: transformed_observation){
      particle.weight *= obs.gausian;
    }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  float beta = 0;
  int index = rand()%num_particles;
  int max_weight = INT_MIN;
  for(auto i: particles){
    weights.push_back(i.weight);
    if(i.weight > max_weight){
      max_weight = i.weight;
    }
  }
  for(int i = 0; i < num_particles; i++){
    beta += rand()%(2*max_weight);
    while(weights[index] < beta){
      beta -= weights[index];
      index = (index+1)%num_particles;
    }
    new_particles.push_back(particles[i]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}