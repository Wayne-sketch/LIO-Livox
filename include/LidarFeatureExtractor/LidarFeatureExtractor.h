#ifndef LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
#define LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
#include <ros/ros.h>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <future>
#include "opencv2/core.hpp"
#include "segment/segment.hpp"
class LidarFeatureExtractor{
    typedef pcl::PointXYZINormal PointType;
public:
    //ctx: lidar特征提取器的构造函数，详细说明在函数定义处
    /** \brief constructor of LidarFeatureExtractor
      * \param[in] n_scans: lines used to extract lidar features
      */
    //ctx: n_scans: lidar扫描线数
    LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,int PartNum,float FlatThreshold,
                          float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis);

    //静态成员函数，可以直接用LiadaFeatureExtractor::_float_as_int调用
    /** \brief transform float to int
      */
    static uint32_t _float_as_int(float f){
      //ctx: 这里定义了一个联合（union）conv，其中包含两个成员变量：一个名为i的32位无符号整数和一个名为f的32位浮点数。
      //ctx: 联合的特点是所有成员共用同一块内存空间，因此在给其中一个成员赋值后，其他成员的值也会被改变。
      //ctx: 注意，这种类型转换可能导致精度损失和不确定的结果，特别是在浮点数表示范围超出整数表示范围的情况下。
      union{uint32_t i; float f;} conv{};
      conv.f = f;
      return conv.i;
    }

    //ctx: 原理同上
    /** \brief transform int to float
      */
    static float _int_as_float(uint32_t i){
      union{float f; uint32_t i;} conv{};
      conv.i = i;
      return conv.f;
    }

    /** \brief Determine whether the point_list is flat
      * \param[in] point_list: points need to be judged
      * \param[in] plane_threshold
      */
    //ctx: 判断点集是否是平面
    bool plane_judge(const std::vector<PointType>& point_list,const int plane_threshold);

    /** \brief Detect lidar feature points
     * ctx: 点云特征检测
      * \param[in] cloud: original lidar cloud need to be detected
      * ctx: 输入雷达点云
      * \param[in] pointsLessSharp: less sharp index of cloud
      * ctx: ?
      * \param[in] pointsLessFlat: less flat index of cloud
      * ctx: ?
      */
    void detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                            std::vector<int>& pointsLessSharp,
                            std::vector<int>& pointsLessFlat);

    void detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                             pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                             pcl::PointCloud<PointType>::Ptr& pointsNonFeature);

    void detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                             std::vector<int>& pointsLessSharp);
                
    //ctx: 点云特征提取
    void FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                     pcl::PointCloud<PointType>::Ptr& laserCloud,
                                     pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                     pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                     pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                     sensor_msgs::PointCloud2 &msg2,
                                     int Used_Line = 1);

    void FeatureExtract_with_segment_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                     	 pcl::PointCloud<PointType>::Ptr& laserCloud,
                                     	 pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                     	 pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                     	 pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                     	 sensor_msgs::PointCloud2 &msg2,
                                     	 int Used_Line = 1);

    /** \brief Detect lidar feature points of CustomMsg
     * ctx:检测CustomMsg的特征点
      * \param[in] msg: original CustomMsg need to be detected
      * ctx: 输入原始CustomMsg
      * \param[in] laserCloud: transform CustomMsg to pcl point cloud format
      * ctx: 将原始数据转化为pcl点云
      * \param[in] laserConerFeature: less Coner features extracted from laserCloud
      * ctx: 从laserCloud中提取的(较少?)Coner特征
      * \param[in] laserSurfFeature: less Surf features extracted from laserCloud
      * ctx: 从laserCloud提取的(较少?)曲面特征
      */
    void FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                        int Used_Line = 1,const int lidar_type=0);

    void FeatureExtract_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                            pcl::PointCloud<PointType>::Ptr& laserCloud,
                            pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                            pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
			    pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                            int Used_Line = 1);
    void FeatureExtract_Mid(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &msg,
                                                   pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                   pcl::PointCloud<PointType>::Ptr& laserSurfFeature);
private:
    //ctx:lidar线数
    /** \brief lines used to extract lidar features */
    const int N_SCANS;

    //ctx: PointType - pcl::PointXYZINormal 保存每条线的原始数据，每个元素都是点云指针，代表一条lidar线上的点云
    /** \brief store original points of each line */
    std::vector<pcl::PointCloud<PointType>::Ptr> vlines;

    //ctx: 保存每条线的角点的索引，第一层vector每个元素代表每条线，第二层vector每个元素代表在同一条线上的角点索引
    //ctx: 索引应该是点云里对应的索引
    /** \brief store corner feature index of each line */
    std::vector<std::vector<int>> vcorner;

    //ctx: 同理，保存的是surf特征点/描述子？
    /** \brief store surf feature index of each line */
    std::vector<std::vector<int>> vsurf;

    int thNumCurvSize;

    float thDistanceFaraway;

    int thNumFlat;
    
    int thPartNum;

    float thFlatThreshold;

    float thBreakCornerDis;

    float thLidarNearestDis;  
};

#endif //LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
