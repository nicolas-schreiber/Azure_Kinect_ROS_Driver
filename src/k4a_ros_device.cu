#include "azure_kinect_ros_driver/k4a_ros_device.h"

#include <thrust/device_vector.h>

struct __attribute__((__packed__)) i16Point {
    int16_t x;
    int16_t y;
    int16_t z;
};

struct __attribute__((__packed__)) ui8RGBA {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
};

/**
 * Point Struct
 * Represents required structure from PointXYZRGB PointCloud ROS Messages
 * Including Padding of the two parts rgb and xyz into two WORDS
 */
struct PointXYZRGB {
  float x = 0;
  float y = 0;
  float z = 0;
  float _pad0;
  float rgb = 0;
  float _pad_1;
  float _pad_2;
  float _pad_3;

  /// Empty Constructor for initial declarations
  __host__ __device__
  inline PointXYZRGB(){};

  /// Constructor from values
  __host__ __device__
  inline PointXYZRGB(float x_, float y_, float z_, float rgb_) 
    : x(x_), y(y_), z(z_), rgb(rgb_) {};

  /// Constructor from individual rgb
  __host__ __device__
  inline PointXYZRGB(float x_, float y_, float z_, uint8_t r, uint8_t g, uint8_t b) : x(x_), y(y_), z(z_)
  {
    uint8_t* rgb_ = (uint8_t*) &rgb;
    rgb_[0] = r;
    rgb_[1] = g;
    rgb_[2] = b;
    rgb_[3] = 0;
  };
};

/**
 * Point Struct
 * Represents required structure from PointXYZ PointCloud ROS Messages
 * Including Padding of the xyz values
 */
struct PointXYZ {
  float x = 0;
  float y = 0;
  float z = 0;
  float _pad;

  /// Empty Constructor for initial declarations
  __host__ __device__
  inline PointXYZ(){};

  /// Constructor from values
  __host__ __device__
  inline PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {};
};



struct RGBDToPointXYZRGB {
  __host__ __device__ 
  PointXYZRGB operator()(i16Point point, ui8RGBA color) {
    // Check Point Validity
    if (point.z <= 0.0f || color.a == 0)
      return PointXYZRGB(nanf(""), nanf(""), nanf(""), 0);
    
    // Create Point
    constexpr float kMillimeterToMeter = 1.0 / 1000.0f;
    float x = kMillimeterToMeter * static_cast<float>(point.x);
    float y = kMillimeterToMeter * static_cast<float>(point.y);
    float z = kMillimeterToMeter * static_cast<float>(point.z);

    return PointXYZRGB(x, y, z, color.r, color.g, color.b);
  }
};

struct DToPointXYZ {
  __host__ __device__ 
  PointXYZ operator()(i16Point point) {
    // Check Point Validity
    if (point.z <= 0.0f)
      return PointXYZ(nanf(""), nanf(""), nanf(""));
    
    // Create Point
    constexpr float kMillimeterToMeter = 1.0 / 1000.0f;
    float x = kMillimeterToMeter * static_cast<float>(point.x);
    float y = kMillimeterToMeter * static_cast<float>(point.y);
    float z = kMillimeterToMeter * static_cast<float>(point.z);

    return PointXYZ(x, y, z);
  }
};


void cudaFillColorPointCloud(const k4a::image& pointcloud_image, const k4a::image& color_image, sensor_msgs::PointCloud2Ptr& point_cloud) 
{
  const size_t point_count = pointcloud_image.get_height_pixels() * pointcloud_image.get_width_pixels();

  const i16Point* h_points = reinterpret_cast<const i16Point*>(pointcloud_image.get_buffer());
  const ui8RGBA*  h_colors = reinterpret_cast<const ui8RGBA*> (color_image.get_buffer());
  PointXYZRGB* h_out = reinterpret_cast<PointXYZRGB*>(point_cloud->data.data());

  thrust::device_vector<i16Point> d_points(point_count);
  thrust::device_vector<ui8RGBA>  d_colors(point_count);
  thrust::device_vector<PointXYZRGB> d_out(point_count);

  thrust::copy(h_points, h_points + point_count, d_points.begin());
  thrust::copy(h_colors, h_colors + point_count, d_colors.begin());

  thrust::transform(d_points.begin(), d_points.end(), d_colors.begin(), d_out.begin(), RGBDToPointXYZRGB());

  thrust::copy(d_out.begin(), d_out.end(), h_out); 
}

void cudaFillPointCloud(const k4a::image& pointcloud_image, sensor_msgs::PointCloud2Ptr& point_cloud) 
{
  const size_t point_count = pointcloud_image.get_height_pixels() * pointcloud_image.get_width_pixels();

  const i16Point* h_points = reinterpret_cast<const i16Point*>(pointcloud_image.get_buffer());
  PointXYZ* h_out = reinterpret_cast<PointXYZ*>(point_cloud->data.data());

  thrust::device_vector<i16Point> d_points(point_count);
  thrust::device_vector<PointXYZ> d_out(point_count);

  thrust::copy(h_points, h_points + point_count, d_points.begin());

  thrust::transform(d_points.begin(), d_points.end(), d_out.begin(), DToPointXYZ());

  thrust::copy(d_out.begin(), d_out.end(), h_out); 
}



