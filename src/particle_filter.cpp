/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */
#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <climits>
#include <numeric>

using std::numeric_limits;
using std::normal_distribution;
using std::string;
using std::vector;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;  // Set the number of particles to be used in the filter
  
  /*
    Creates a normal (Gaussian) distribution for particles' initial x, y, and theta based
    on gps data and measurement noise
  */
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
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
    // This line creates a normal (Gaussian) distribution for the noise predictions
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

// Predict new position based on past position, velocity, time, and yaw_rate data
  for (auto& particle : particles){
    double theta = particle.theta;
    if(fabs(yaw_rate)  < 0.00001){
      particle.x += velocity * delta_t * cos(theta);
      particle.y += velocity * delta_t* sin(theta);
    }else{
      particle.x += (velocity/yaw_rate)*(sin(theta + yaw_rate*delta_t) - sin(theta));
      particle.y += (velocity/yaw_rate)*(-cos(theta + yaw_rate*delta_t) + cos(theta));
      particle.theta += yaw_rate*delta_t;
    }

    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }
}
// Transform observation data from car coordinates to map coordinates
LandmarkObs transformation(int id, float x_particle, float y_particle, float heading_particle, float x_obs, float y_obs){
    float teta = heading_particle;
    LandmarkObs return_value;
    return_value.x = x_particle + (cos(teta) * x_obs) - (sin(teta)* y_obs);
    return_value.y = y_particle + (sin(teta) * x_obs) + (cos(teta) * y_obs);
    return_value.id = id;
    return return_value;

}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                    vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  // For each particle we update its weight
  for (auto& particle: particles){
    vector<LandmarkObs> transformed_observation;
    double gaussian = 1.0;
    for(auto obs: observations){
      LandmarkObs transformed_observation = transformation(obs.id, particle.x, particle.y, particle.theta, obs.x, obs.y);
      vector<double> dist_from_particle;
      double min_distance = numeric_limits<double>::max();
      float prx = 0;
      float pry = 0;
      for(auto& landmark: map_landmarks.landmark_list){
        if(sqrt(pow(landmark.x_f - particle.x, 2) + pow(landmark.y_f - particle.y, 2)) <= sensor_range){
          double dist = sqrt(pow(transformed_observation.x - landmark.x_f, 2) + pow(transformed_observation.y - landmark.y_f, 2));
          if(dist < min_distance) {
            min_distance = dist;
            prx = landmark.x_f;
            pry = landmark.y_f;
          }
          dist_from_particle.push_back(dist);
        }else{
          dist_from_particle.push_back(100000);
        }
      }
      gaussian *= find_gaussian(prx, pry, transformed_observation.x, transformed_observation.y, std_landmark[0], std_landmark[1]);
    }
    particle.weight = gaussian;
    weights.push_back(gaussian);
  }
}
 
void ParticleFilter::resample() {
  // choose new particles based on their current weights
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    std::discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles.push_back(particles[index(gen)]);
  }
  particles = new_particles;
  weights.clear();
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