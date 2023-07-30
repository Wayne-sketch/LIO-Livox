#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

//ctx: 构造函数，初始化列表 9个参数
/*
1、n_scans: int，lidar扫描线数
2、NumCurvSize: int，表示曲率点的数量 ？
3、DistanceFaraway: float，表示远处的距离。
4、NumFlat: int，表示平坦点的数量。
5、PartNum: int，表示对点云分割区域的数量。
6、FlatThreshold: float，表示平坦度的阈值。
7、BreakCornerDis: float，表示打破角点的距离。
8、LidarNearestDis: float，表示激光雷达最近点的距离。
9、KdTreeCornerOutlierDis: float，表示Kd树角点的离群值距离。没用到？*/
LidarFeatureExtractor::LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,
                                             int PartNum,float FlatThreshold,float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis)
                                             :N_SCANS(n_scans),//lidar扫描线数
                                              thNumCurvSize(NumCurvSize),//thNumCurvSize代表计算曲率时，被计算点两端要分别采用几个点
                                              thDistanceFaraway(DistanceFaraway),//判断是否为远点的阈值
                                              thNumFlat(NumFlat),//一个区域内最大允许的的平面点数量，也有可能选出来的平面点数量比thNumFlat大
                                              thPartNum(PartNum),
                                              thFlatThreshold(FlatThreshold),
                                              thBreakCornerDis(BreakCornerDis),
                                              thLidarNearestDis(LidarNearestDis){
  //为vlines、vcorner和vsurf这三个成员变量分配内存空间。
  //原始点云数据，vector<点云指针>    vlines.resize(N_SCANS) 将 vlines 初始化为包含 N_SCANS 个元素的空的vector。
  //ctx: PointType - pcl::PointXYZINormal 保存每条线的原始数据，每个元素都是点云指针，代表一条lidar线上的点云
  vlines.resize(N_SCANS);
  //遍历点云
  for(auto & ptr : vlines){
    ptr.reset(new pcl::PointCloud<PointType>());
  }
  //保存每条线的角点的索引 vector<vector<int>>
  vcorner.resize(N_SCANS);
  //保存每条线的surf特征点的索引 vector<vector<int>>s
  vsurf.resize(N_SCANS);
}


/*原理：
在点云数据中，如果所有点都位于一个平面上，则该平面可以用一个二维平面表示。在二维平面中，只需要两个坐标轴（通常是X和Y轴）来表示所有点的位置。这意味着在这个平面上，
所有点的运动方向或分布主要集中在这两个坐标轴上，而在垂直于这个平面的方向上的运动或分布应该相对较小。

奇异值分解（Singular Value Decomposition，SVD）是一种矩阵分解技术，可以将一个矩阵分解成三个矩阵的乘积：A = U * S * V^T。其中，U和V是正交矩阵，S是一个对角矩阵，对角线上的元素称为奇异值。
在这里，我们通过奇异值分解对计算的平均平方误差矩阵进行分解。平均平方误差矩阵描述了点云数据在三个坐标轴（X、Y、Z）方向上的离散程度。若点云在一个平面上，它们的运动主要集中在平面上的两个方向，
即Z轴方向上的方差较小。

当我们比较第一个奇异值 _matD1(0, 0) 和第二个奇异值 _matD1(1, 0) 的大小时，如果第一个奇异值远小于第二个奇异值，即 _matD1(0, 0) < plane_threshold * _matD1(1, 0) 条件成立，
那么可以认为数据在Z轴方向上的方差较小，而主要集中在一个平面上。这是因为如果数据在一个平面上，则在Z轴方向上的运动或分布相对较小，导致第一个奇异值较小。

因此，通过比较奇异值，我们可以得出点云数据是否主要位于一个平面上的结论。请注意，plane_threshold 是一个阈值参数，用于控制判断的敏感度，其具体取值根据具体应用和点云数据的特性进行调整。*/

//判断给定点云列表是否表示一个平面
bool LidarFeatureExtractor::plane_judge(const std::vector<PointType>& point_list,const int plane_threshold)
{
  //计算给定点云列表 point_list 中所有点的平均坐标 (cx, cy, cz)。
  int num = point_list.size();
  float cx = 0;
  float cy = 0;
  float cz = 0;
  for (int j = 0; j < num; j++) {
    cx += point_list[j].x;
    cy += point_list[j].y;
    cz += point_list[j].z;
  }
  //将结果存储在 cx、cy 和 cz 变量中
  cx /= num;
  cy /= num;
  cz /= num;

  //这部分代码计算了点云列表中点与平均点坐标之间的差值，并用这些差值计算平均平方误差矩阵。
  //该平方误差矩阵是一个3x3的矩阵，其中每个元素表示点与平均点之间的偏差。这些偏差后来被用来判断点云列表是否表示一个平面。
  //mean square error
  float a11 = 0;
  float a12 = 0;
  float a13 = 0;
  float a22 = 0;
  float a23 = 0;
  float a33 = 0;
  //遍历点云
  for (int j = 0; j < num; j++) {
    float ax = point_list[j].x - cx;
    float ay = point_list[j].y - cy;
    float az = point_list[j].z - cz;
    //计算每个点到平均点差值的平方和
    a11 += ax * ax;
    a12 += ax * ay;
    a13 += ax * az;
    a22 += ay * ay;
    a23 += ay * az;
    a33 += az * az;
  }
  //平均平方误差值
  a11 /= num;
  a12 /= num;
  a13 /= num;
  a22 /= num;
  a23 /= num;
  a33 /= num;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();
  Eigen::Matrix< double, 3, 1 > _matD1;
  _matD1.setZero();
  Eigen::Matrix< double, 3, 3 > _matV1;
  _matV1.setZero();

  //matA1对称矩阵 协方差矩阵
  _matA1(0, 0) = a11;
  _matA1(0, 1) = a12;
  _matA1(0, 2) = a13;
  _matA1(1, 0) = a12;
  _matA1(1, 1) = a22;
  _matA1(1, 2) = a23;
  _matA1(2, 0) = a13;
  _matA1(2, 1) = a23;
  _matA1(2, 2) = a33;

  //使用JacobiSVD（奇异值分解）对 _matA1 进行分解，得到奇异值和左奇异向量矩阵。
  //左奇异向量是原始矩阵的列向量，而右奇异向量是原始矩阵的行向量。
  /* 其他可能的设置参数(未实验验证)
  Eigen::ComputeFullU and Eigen::ComputeFullV:
  这两个选项用于计算完整的左奇异向量矩阵和右奇异向量矩阵。与之前的Thin选项不同，这些选项将计算完整的奇异向量矩阵，不进行截断，输出完整的矩阵。

  Eigen::AllowOutOfCoreComputation:
  这个选项用于允许JacobiSVD在内存不足时，使用外存储器来计算奇异值分解。当处理非常大的矩阵时，可能会遇到内存不足的问题，设置这个选项可以帮助解决这个问题。

  Eigen::ComputeFullU | Eigen::ComputeFullV:
  这是一个组合选项，用于同时计算完整的左奇异向量矩阵和右奇异向量矩阵。

  Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::FullPivHouseholderQRPreconditioner:
  这也是一个组合选项，用于计算完整的左奇异向量矩阵和右奇异向量矩阵，并使用FullPivHouseholderQR预处理器来优化计算。

  Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::ComputeFullU | Eigen::ComputeFullV:
  这是一个非常完整的组合选项，可以同时计算所有可能的结果，包括Thin和Full的左右奇异向量矩阵。*/
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(_matA1, Eigen::ComputeThinU | Eigen::ComputeThinV);
  _matD1 = svd.singularValues();
  _matV1 = svd.matrixU();
  //判断点云列表是否表示一个平面
  //SVD分解找到主要分布方向
  //它比较第一个奇异值和第二个奇异值的大小，如果第一个奇异值远远小于第二个奇异值的 plane_threshold 倍，就返回 true 表示点云列表表示一个平面，否则返回 false 表示不是平面。
  if (_matD1(0, 0) < plane_threshold * _matD1(1, 0)) {
    return true;
  }
  else{
    return false;
  }
}

//点云特征提取
//第一个参数是点云指针的引用，可能要修改点云指针，提取的特征点存储在两个vector中
//该算法旨在检测三种类型的特征点：尖锐的拐角、平坦的表面以及表面相遇处的断点。
//输出：识别出的尖锐拐角点被添加到laserCloudCorner点云中，并将尖锐拐角点和平坦表面点的索引添加到pointsLessSharp和pointsLessFlat向量中。
void LidarFeatureExtractor::detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp,
                                                std::vector<int>& pointsLessFlat){
  /*特征标志：使用固定大小的数组CloudFeatureFlag来存储每个点的标志。标志的值用于将每个点归类如下：
  0：未处理的点（初始值为0）
  1：表示该点前后2or3个点中有3；
  2：曲率最小的几个被标为3的点，或被标为3的远点，或点与前后点的角度小于15°的点；最终存为平面点
  3：所有曲率小于阈值的点，注意：一部分3点后面会变成2点；
  300：强度变化大于阈值且曲率很小的3个点
  150：两个面的夹角构成的边缘点
  100：断点，左右相邻点距离差超过1m,扫描线与邻点向量夹角大于18 100和150的为角点*/

  //用于存储每个点的特征标志的数组，初始时都设置为0。
  int CloudFeatureFlag[20000];

  //用于存储每个点的曲率值的数组。
  float cloudCurvature[20000];

  //用于存储每个点到激光雷达的距离的数组。
  float cloudDepth[20000];

  //用于存储点云点的索引，将会根据曲率值从小到大进行排序。
  int cloudSortInd[20000];

  //用于存储每个点的反射强度的数组。
  float cloudReflect[20000];

  //用于存储点云点的索引，将会根据反射强度从小到大进行排序。
  int reflectSortInd[20000];

  //如果点与前后点的角度均小于15°，将该点标志置为1
  int cloudAngle[20000];

  //给传进来的点云指针创建一个引用
  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;
  //获取输入点云的大小
  int cloudSize = laserCloudIn->points.size();
  //临时变量point 一个点
  PointType point;
  //定义一个新的点云对象_laserCloud，用于存储去除非法点后的点云，非法点指的是三位坐标含有NaN Inf的值，下面马上就能看出来
  //new出一个点云，()的意思是调用默认构造函数，返回一个点云指针，作为pcl::PointCloud<PointType>::Ptr类构造函数的参数 猜测pcl::PointCloud<PointType>::Ptr类是智能指针类
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  //预先分配空间，以避免在添加点时不断重新分配内存。
  _laserCloud->reserve(cloudSize);

  //遍历点云中的点
  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
//条件编译，如果定义了UNDISTORT宏，则执行#ifdef UNDISTORT和#endif之间的代码，否则执行#else和#endif之间的代码。
//如果UNDISTORT被定义，则获取点云中第i个点的法线normal_x。如果未定义UNDISTORT，则将点的法线normal_x设置为1.0（默认值）。
#ifdef UNDISTORT
    point.normal_x = laserCloudIn.points[i].normal_x;
#else
    point.normal_x = 1.0;
#endif

    //获取点云中第i个点的反射强度。
    point.intensity = laserCloudIn->points[i].intensity;
    //如果点的x、y或z坐标中存在非有限值（例如NaN或Inf），则跳过当前点，不加入新的点云对象。
    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }
    //将新的点point加入_laserCloud点云对象中
    _laserCloud->push_back(point);
    //为当前点设置初始特征标志为0（未处理）。
    CloudFeatureFlag[i] = 0;
  }//点云中的点遍历完成一次

  //点云中的点遍历完成后，更新cloudSize，表示去除非法点后的点云大小。，后面点云大小用更新后的，点云就用去掉非法点后的点云
  cloudSize = _laserCloud->size();

  //定义并初始化用于调试的计数器变量。作用后面会看到，这里需要标一下？
  int debugnum1 = 0;
  int debugnum2 = 0;
  int debugnum3 = 0;
  int debugnum4 = 0;
  int debugnum5 = 0;

  //用于控制点云遍历时的步长，默认为1。
  int count_num = 1;
  //用于标记一个点左右侧是否是平面
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //---------------------------------------- surf feature extract ---------------------------------------------
  /*表面特征提取（surf feature extract）：代码迭代遍历点云以检测平坦表面点。它计算每个点及其邻近点的曲率，以确定是否属于平坦表面。该算法考虑点的深度和邻近点之间的角度来分类表面点。*/
  //定义了点云处理的起始和结束索引，排除了前5个点和后5个点。

  /*
  *
  *
  * 对于平面点的特征有四种标志为1，2，3，300，平面特征提取把一条扫描线分成了100段进行提取，最终选取标志位为2的点为平面点
  * 
  * 
  * 
  * */
  int scanStartInd = 5;
  int scanEndInd = cloudSize - 6;

  //用于记录距离过远的点的数量，作为特征的一部分。？？
  int thDistanceFaraway_fea = 0;
  //遍历从第6个点到倒数第6个点的点云数据。
  for (int i = 5; i < cloudSize - 5; i ++ ) {
    //计算坐标差值
    float diffX = 0;
    float diffY = 0;
    float diffZ = 0;
    
    //计算点的距离（深度）：通过sqrt函数计算点到原点的距离。 _laserCloud是去除非法点后的点云
    float dis = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                     _laserCloud->points[i].y * _laserCloud->points[i].y +
                     _laserCloud->points[i].z * _laserCloud->points[i].z);
    //定义了三个Eigen::Vector3d对象，分别存储当前点、前一个点和后一个点的(x, y, z)坐标。
    Eigen::Vector3d pt_last(_laserCloud->points[i-1].x, _laserCloud->points[i-1].y, _laserCloud->points[i-1].z);
    Eigen::Vector3d pt_cur(_laserCloud->points[i].x, _laserCloud->points[i].y, _laserCloud->points[i].z);
    Eigen::Vector3d pt_next(_laserCloud->points[i+1].x, _laserCloud->points[i+1].y, _laserCloud->points[i+1].z);
    //计算当前点与前后两个点的夹角余弦值。余弦值越大才越有可能是平坦表面点？可是pt_cur并不是点的法向量
    double angle_last = (pt_last-pt_cur).dot(pt_cur) / ((pt_last-pt_cur).norm()*pt_cur.norm());
    double angle_next = (pt_next-pt_cur).dot(pt_cur) / ((pt_next-pt_cur).norm()*pt_cur.norm());

    //根据距离和夹角值判断曲率点的数量，设置thNumCurvSize的值为2或3，thNumCurvSize代表计算曲率时，被计算点两端要分别采用几个点，为什么要这样的策略来取点数？
    //   &&优先级大于||
    if (dis > thDistanceFaraway || (fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966)) {
      thNumCurvSize = 2;
    } else {
      thNumCurvSize = 3;
    }

    if(fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966) {
      //如果当前点的夹角余弦值的绝对值较大，将对应的cloudAngle标志设置为1。
      cloudAngle[i] = 1;
    }

    //计算特征值：通过对当前点前后若干个点的坐标差值，计算得到该点的曲率值、深度、反射强度等特征
    //初始化变量diffR，用于存储特征值中的反射强度信息。 反射强度信息的意义？
    float diffR = -2 * thNumCurvSize * _laserCloud->points[i].intensity;
    //循环计算特征值：通过对当前点前后若干个点的坐标差值，计算得到该点的曲率值和反射强度。
    for (int j = 1; j <= thNumCurvSize; ++j) {
      //这里还没有算差值，只是把坐标值累计起来
      //在每次循环中，累加了坐标的差值diffX、diffY、diffZ以及反射强度差值diffR。
      diffX += _laserCloud->points[i - j].x + _laserCloud->points[i + j].x;
      diffY += _laserCloud->points[i - j].y + _laserCloud->points[i + j].y;
      diffZ += _laserCloud->points[i - j].z + _laserCloud->points[i + j].z;
      diffR += _laserCloud->points[i - j].intensity + _laserCloud->points[i + j].intensity;
    }
    //这里才一起算了差值
    //修正之前累加的坐标差值，减去当前点坐标的2倍，用于计算曲率。
    diffX -= 2 * thNumCurvSize * _laserCloud->points[i].x;
    diffY -= 2 * thNumCurvSize * _laserCloud->points[i].y;
    diffZ -= 2 * thNumCurvSize * _laserCloud->points[i].z;

    //将特征值存储：将计算得到的特征值存储在对应的数组cloudCurvature、cloudDepth、cloudSortInd、cloudReflect和reflectSortInd中。
    cloudDepth[i] = dis;//存储深度信息
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;// / (2 * thNumCurvSize * dis + 1e-3); 存储曲率信息
    cloudSortInd[i] = i;// 存储排序索引，用于后续对曲率和反射强度进行排序
    cloudReflect[i] = diffR;// 存储反射强度信息
    reflectSortInd[i] = i;// 存储反射强度排序索引，用于后续对反射强度进行排序

  }//结束遍历从第6个点到倒数第6个点的点云数据。

  //循环遍历thPartNum个区域：该部分代码对每个区域内的点进行特征标记和提取。
  for (int j = 0; j < thPartNum; j++) {
    //在这个循环中，首先计算每个区域的起始点索引sp和结束点索引ep，通过将整个点云分成thPartNum个区域，以便逐个区域处理点云数据。
    //scanStarInd是从第6个点云开始
    int sp = scanStartInd + (scanEndInd - scanStartInd) * j / thPartNum;
    //-1是因为要把下一段的开始点去掉，但是会导致没有算最后一个点？
    int ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / thPartNum - 1;

    //对区域内的点进行冒泡排序，分别根据曲率值和反射强度从小到大进行排序，以便后续特征提取过程中选择合适的特征点。
    // sort the curvatures from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudCurvature[cloudSortInd[l]] <
            cloudCurvature[cloudSortInd[l - 1]]) {
          //对存储索引的vector排序，注意只对索引排序而不是对曲率或反射强度排序
          int temp = cloudSortInd[l - 1];
          cloudSortInd[l - 1] = cloudSortInd[l];
          cloudSortInd[l] = temp;
        }
      }
    }

    // sort the reflectivity from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudReflect[reflectSortInd[l]] <
            cloudReflect[reflectSortInd[l - 1]]) {
          int temp = reflectSortInd[l - 1];
          reflectSortInd[l - 1] = reflectSortInd[l];
          reflectSortInd[l] = temp;
        }
      }
    }

    //记录选取的特征点数量：用于记录在当前区域内已经选取的最平坦点的数量和最尖锐点的数量。
    int smallestPickedNum = 1;
    int sharpestPickedNum = 1;
    //特征点标记和提取：根据一定的条件，对当前区域内的点进行特征标记和提取。
    for (int k = sp; k <= ep; k++) {
      //从曲率值小的索引开始遍历本区域
      int ind = cloudSortInd[k];
      //如果该点已经被标记为特征点，则跳过
      if (CloudFeatureFlag[ind] != 0) continue;
      
      // 如果当前点的曲率值小于一定阈值，则认为该点是平坦区域的一部分
      if (cloudCurvature[ind] < thFlatThreshold * cloudDepth[ind] * thFlatThreshold * cloudDepth[ind]) {
        //将该点标记为3
        CloudFeatureFlag[ind] = 3;

        //向后查找一定数量的点，如果这些点的坐标变化不大，则将这些点也标记为平坦特征点（待定）
        for (int l = 1; l <= thNumCurvSize; l++) {
          //前后点之间的坐标差值
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l - 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l - 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l - 1].z;
          // 如果坐标变化过大或者当前点的深度超过一定阈值，则停止向前查找
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }
          // 标记为3的点后面的点标记为1
          CloudFeatureFlag[ind + l] = 1;
        }
        //向前查找一定数量的点，如果这些点的坐标变化不大，则将这些点也标记为平坦特征点（待定）
        for (int l = -1; l >= -thNumCurvSize; l--) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l + 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l + 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l + 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }

          CloudFeatureFlag[ind + l] = 1;
        }
      }
    }//在这个循环中，对当前区域内的点进行遍历，判断是否满足平坦特征的条件。如果满足条件，将该点及其周围的若干点标记为平坦特征点（待定）。这里使用了一定的坐标变化阈值和深度阈值来控制点的选择。
    
    for (int k = sp; k <= ep; k++) {
      //从曲率值小的索引开始遍历本区域
      int ind = cloudSortInd[k];
      //如果该点是最平坦的点，并且在规定的最大平坦点数量内，则将其标记为普通特征点
      //曲率最小的几个被标为3的点，或被标为3的远点，或点与前后点的角度小于15°的点；最终存为平面点
      if(((CloudFeatureFlag[ind] == 3) && (smallestPickedNum <= thNumFlat)) || 
          ((CloudFeatureFlag[ind] == 3) && (cloudDepth[ind] > thDistanceFaraway)) ||
          cloudAngle[ind] == 1){
        smallestPickedNum ++;
        CloudFeatureFlag[ind] = 2;
        // 如果该点的深度超过一定阈值，则增加计数器
        if(cloudDepth[ind] > thDistanceFaraway) {
          thDistanceFaraway_fea++;
        }
      }

      // 如果该点是最尖锐的点，并且在规定的最大尖锐点数量内，并且反射强度满足一定条件，则将其标记为特殊特征点？？
      //从反射强度最小的索引开始遍历本区域
      //300——强度变化大于阈值且曲率很小的3个点
      int idx = reflectSortInd[k];
      if(cloudCurvature[idx] < 0.7 * thFlatThreshold * cloudDepth[idx] * thFlatThreshold * cloudDepth[idx]
         && sharpestPickedNum <= 3 && cloudReflect[idx] > 20.0){
        sharpestPickedNum ++;
        CloudFeatureFlag[idx] = 300;
      }
    }
    
  }//结束循环遍历thPartNum个区域：该部分代码对每个区域内的点进行特征标记和提取。

  //---------------------------------------- line feature where surfaces meet -------------------------------------
  /*线特征提取（line feature where surfaces meet）：代码进一步处理点，以确定表面相遇处的断点。它查找左右曲率明显不同的点，并计算相邻点法线之间的夹角。如果角度低于阈值，则将该点标记为断点。*/
  for (int i = 5; i < cloudSize - 5; i += count_num ) {
    //计算当前点 (x, y, z) 到原点 (0, 0, 0) 的欧几里得距离，即点的深度
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);
    //left curvature
    //计算左曲率：计算左侧曲率的平方，通过计算当前点和其前四个相邻点之间的差值来得到。这里涉及到坐标差值的计算，并将差值平方后累加得到左曲率值，存储在变量 left_curvature 中。
    float ldiffX =
            _laserCloud->points[i - 4].x + _laserCloud->points[i - 3].x
            - 4 * _laserCloud->points[i - 2].x
            + _laserCloud->points[i - 1].x + _laserCloud->points[i].x;

    float ldiffY =
            _laserCloud->points[i - 4].y + _laserCloud->points[i - 3].y
            - 4 * _laserCloud->points[i - 2].y
            + _laserCloud->points[i - 1].y + _laserCloud->points[i].y;

    float ldiffZ =
            _laserCloud->points[i - 4].z + _laserCloud->points[i - 3].z
            - 4 * _laserCloud->points[i - 2].z
            + _laserCloud->points[i - 1].z + _laserCloud->points[i].z;

    float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

    //深度越大，给的范围就越大
    if(left_curvature < thFlatThreshold * depth){

      std::vector<PointType> left_list;
      //如果认为是平面点，存到left_list中
      for(int j = -4; j < 0; j++){
        left_list.push_back(_laserCloud->points[i + j]);
      }
      //把左侧标记为平面
      left_surf_flag = true;
    }
    else{
      //把左侧标记为非平面
      left_surf_flag = false;
    }

    //right curvature
    //右侧同理
    float rdiffX =
            _laserCloud->points[i + 4].x + _laserCloud->points[i + 3].x
            - 4 * _laserCloud->points[i + 2].x
            + _laserCloud->points[i + 1].x + _laserCloud->points[i].x;

    float rdiffY =
            _laserCloud->points[i + 4].y + _laserCloud->points[i + 3].y
            - 4 * _laserCloud->points[i + 2].y
            + _laserCloud->points[i + 1].y + _laserCloud->points[i].y;

    float rdiffZ =
            _laserCloud->points[i + 4].z + _laserCloud->points[i + 3].z
            - 4 * _laserCloud->points[i + 2].z
            + _laserCloud->points[i + 1].z + _laserCloud->points[i].z;

    float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

    if(right_curvature < thFlatThreshold * depth){
      std::vector<PointType> right_list;
      //如果判断右侧是平面，把右侧四个点存入right_list
      for(int j = 1; j < 5; j++){
        right_list.push_back(_laserCloud->points[i + j]);
      }
      //点云遍历步长改为4，跳过下面四个点的遍历，同时标记右侧为平面
      count_num = 4;
      right_surf_flag = true;
    }
    else{
      //否则点云遍历步长还是1，标记右侧为非平面点
      count_num = 1;
      right_surf_flag = false;
    }

    //calculate the included angle
    if(left_surf_flag && right_surf_flag){
      debugnum4 ++;

      //初始化两个 Eigen::Vector3d 对象 norm_left 和 norm_right，分别用于计算左侧和右侧特征组的法向量。应该不是法向量，不知道怎么命名？？？
      Eigen::Vector3d norm_left(0,0,0);
      Eigen::Vector3d norm_right(0,0,0);
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_left += (k/10.0)* tmp;
      }
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_right += (k/10.0)* tmp;
      }

      //calculate the angle between this group and the previous group
      double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
      //calculate the maximum distance, the distance cannot be too small
      Eigen::Vector3d last_tmp = Eigen::Vector3d(_laserCloud->points[i - 4].x - _laserCloud->points[i].x,
                                                 _laserCloud->points[i - 4].y - _laserCloud->points[i].y,
                                                 _laserCloud->points[i - 4].z - _laserCloud->points[i].z);
      Eigen::Vector3d current_tmp = Eigen::Vector3d(_laserCloud->points[i + 4].x - _laserCloud->points[i].x,
                                                    _laserCloud->points[i + 4].y - _laserCloud->points[i].y,
                                                    _laserCloud->points[i + 4].z - _laserCloud->points[i].z);
      double last_dis = last_tmp.norm();
      double current_dis = current_tmp.norm();

      //150——两个面的夹角构成的边缘点
      if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
        debugnum5 ++;
        CloudFeatureFlag[i] = 150;
      }
    }

  }

  //--------------------------------------------------- break points ---------------------------------------------
  /*角点特征提取（break points）：代码通过比较前后方向上相邻点之间的距离差异来识别尖锐的拐角。如果差异超过阈值，则将该点视为尖锐拐角。*/
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);
    
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool left_is_plane = plane_judge(left_list,100);

        if( cc < 0.95 ){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool right_is_plane = plane_judge(right_list,100);

        if( cc < 0.95){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      debugnum2++;
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      if(cc < 0.95){
        debugnum3++;
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  //push_back feature

  for(int i = 5; i < cloudSize - 5; i ++){

    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x
            + _laserCloud->points[i].y * _laserCloud->points[i].y
            + _laserCloud->points[i].z * _laserCloud->points[i].z;

    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

    if(CloudFeatureFlag[i] == 2){
      pointsLessFlat.push_back(i);
      num_surf++;
      continue;
    }

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }

  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}

void LidarFeatureExtractor::FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                        sensor_msgs::PointCloud2 &msg_seg,
                                                        const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  int *idtrans = (int*)calloc(dnum, sizeof(int));
  float *data=(float*)calloc(dnum*4,sizeof(float));
  int point_num = 0;

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }

  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
    int line_idx = _float_as_int(laserCloud->points[i].normal_y);
    laserCloud->points[i].normal_z = _int_as_float(i);
    vlines[line_idx]->push_back(laserCloud->points[i]);
  }

  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
    threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint3, this, std::ref(vlines[i]),std::ref(vcorner[i]));
  }

  for(int i=0; i<N_SCANS; ++i){
    threads[i].join();
  }

  int num_corner = 0;
  for(int i=0; i<N_SCANS; ++i){
    for(int j=0; j<vcorner[i].size(); ++j){
      laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0; 
      num_corner++;
    }
  }

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}

void LidarFeatureExtractor::FeatureExtract_with_segment_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                            pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                            pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                            sensor_msgs::PointCloud2 &msg_seg,
                                                            const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  int *idtrans = (int*)calloc(dnum, sizeof(int));
  float *data=(float*)calloc(dnum*4,sizeof(float));
  int point_num = 0;

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }

  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  std::size_t cloud_num = laserCloud->size();

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}


void LidarFeatureExtractor::detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                                                pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                                                pcl::PointCloud<PointType>::Ptr& pointsNonFeature){

  int cloudSize = cloud->points.size();

  pointsLessFlat.reset(new pcl::PointCloud<PointType>());
  pointsNonFeature.reset(new pcl::PointCloud<PointType>());

  pcl::KdTreeFLANN<PointType>::Ptr KdTreeCloud;
  KdTreeCloud.reset(new pcl::KdTreeFLANN<PointType>);
  KdTreeCloud->setInputCloud(cloud);

  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;

  int num_near = 10;
  int stride = 1;
  int interval = 4;

  for(int i = 5; i < cloudSize - 5; i = i+stride) {
    if(fabs(cloud->points[i].normal_z - 1.0) < 1e-5) {
      continue;
    }

    double thre1d = 0.5;
    double thre2d = 0.8;
    double thre3d = 0.5;
    double thre3d2 = 0.13;

    double disti = sqrt(cloud->points[i].x * cloud->points[i].x + 
                        cloud->points[i].y * cloud->points[i].y + 
                        cloud->points[i].z * cloud->points[i].z);

    if(disti < 30.0) {
      thre1d = 0.5;
      thre2d = 0.8;
      thre3d2 = 0.07;
      stride = 14;
      interval = 4;
    } else if(disti < 60.0) {
      stride = 10;
      interval = 3;
    } else {
      stride = 1;
      interval = 0;
    }

    if(disti > 100.0) {
      num_near = 6;

      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
      continue;
    } else if(disti > 60.0) {
      num_near = 8;
    } else {
      num_near = 10;
    }

    KdTreeCloud->nearestKSearch(cloud->points[i], num_near, _pointSearchInd, _pointSearchSqDis);

    if (_pointSearchSqDis[num_near-1] > 5.0 && disti < 90.0) {
      continue;
    }

    Eigen::Matrix< double, 3, 3 > _matA1;
    _matA1.setZero();

    float cx = 0;
    float cy = 0;
    float cz = 0;
    for (int j = 0; j < num_near; j++) {
      cx += cloud->points[_pointSearchInd[j]].x;
      cy += cloud->points[_pointSearchInd[j]].y;
      cz += cloud->points[_pointSearchInd[j]].z;
    }
    cx /= num_near;
    cy /= num_near;
    cz /= num_near;

    float a11 = 0;
    float a12 = 0;
    float a13 = 0;
    float a22 = 0;
    float a23 = 0;
    float a33 = 0;
    for (int j = 0; j < num_near; j++) {
      float ax = cloud->points[_pointSearchInd[j]].x - cx;
      float ay = cloud->points[_pointSearchInd[j]].y - cy;
      float az = cloud->points[_pointSearchInd[j]].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }
    a11 /= num_near;
    a12 /= num_near;
    a13 /= num_near;
    a22 /= num_near;
    a23 /= num_near;
    a33 /= num_near;

    _matA1(0, 0) = a11;
    _matA1(0, 1) = a12;
    _matA1(0, 2) = a13;
    _matA1(1, 0) = a12;
    _matA1(1, 1) = a22;
    _matA1(1, 2) = a23;
    _matA1(2, 0) = a13;
    _matA1(2, 1) = a23;
    _matA1(2, 2) = a33;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
    double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) / sqrt(saes.eigenvalues()[2]);
    double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) / sqrt(saes.eigenvalues()[2]);
    double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

    if(a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 2.0;
      pointsLessFlat->points.push_back(cloud->points[i]);
    } else if(a3d > thre3d) {
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
    }
  }  
}


void LidarFeatureExtractor::detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp){
  int CloudFeatureFlag[20000];
  float cloudCurvature[20000];
  float cloudDepth[20000];
  int cloudSortInd[20000];
  float cloudReflect[20000];
  int reflectSortInd[20000];
  int cloudAngle[20000];

  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;

  int cloudSize = laserCloudIn->points.size();

  PointType point;
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
    point.normal_x = 1.0;
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }

  cloudSize = _laserCloud->size();

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);

    
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool left_is_plane = plane_judge(left_list,0.3);

        if(cc < 0.93){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool right_is_plane = plane_judge(right_list,0.3);

        if(cc < 0.93){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      if(cc < 0.93){
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  for(int i = 5; i < cloudSize - 5; i ++){
    Eigen::Vector3d left_pt = Eigen::Vector3d(_laserCloud->points[i - 1].x,
                                              _laserCloud->points[i - 1].y,
                                              _laserCloud->points[i - 1].z);
    Eigen::Vector3d right_pt = Eigen::Vector3d(_laserCloud->points[i + 1].x,
                                               _laserCloud->points[i + 1].y,
                                               _laserCloud->points[i + 1].z);

    Eigen::Vector3d cur_pt = Eigen::Vector3d(_laserCloud->points[i].x,
                                             _laserCloud->points[i].y,
                                             _laserCloud->points[i].z);

    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x +
                _laserCloud->points[i].y * _laserCloud->points[i].y +
                _laserCloud->points[i].z * _laserCloud->points[i].z;

    double clr = fabs(left_pt.dot(right_pt) / (left_pt.norm()*right_pt.norm()));
    double cl = fabs(left_pt.dot(cur_pt) / (left_pt.norm()*cur_pt.norm()));
    double cr = fabs(right_pt.dot(cur_pt) / (right_pt.norm()*cur_pt.norm()));

    if(clr < 0.999){
      CloudFeatureFlag[i] = 200;
    }

    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 200){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }
  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}


void LidarFeatureExtractor::FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserCloud,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                           const int Used_Line,const int lidar_type){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
  ptr->clear();
  }
  for(auto & v : vcorner){
  v.clear();
  }
  for(auto & v : vsurf){
  v.clear();
  }
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
  int line_num = (int)p.line;
  if(line_num > Used_Line-1) continue;
  if(lidar_type == 0||lidar_type == 1)
  {
      if(p.x < 0.01) continue;
  }
  else if(lidar_type == 2)
  {
      if(std::fabs(p.x) < 0.01) continue;
  }
//  if(p.x < 0.01) continue;
  point.x = p.x;
  point.y = p.y;
  point.z = p.z;
  point.intensity = p.reflectivity;
  point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
  point.normal_y = _int_as_float(line_num);
  laserCloud->push_back(point);
  }
  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
  int line_idx = _float_as_int(laserCloud->points[i].normal_y);
  laserCloud->points[i].normal_z = _int_as_float(i);
  vlines[line_idx]->push_back(laserCloud->points[i]);
  laserCloud->points[i].normal_z = 0;
  }
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
  threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                     std::ref(vcorner[i]), std::ref(vsurf[i]));
  }
  for(int i=0; i<N_SCANS; ++i){
  threads[i].join();
  }
  for(int i=0; i<N_SCANS; ++i){
  for(int j=0; j<vcorner[i].size(); ++j){
  laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
  }
  for(int j=0; j<vsurf[i].size(); ++j){
  laserCloud->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
  }
  }

  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 1.0) < 1e-5)
  laserConerFeature->push_back(p);
  }
  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 2.0) < 1e-5)
  laserSurfFeature->push_back(p);
  }
}

void LidarFeatureExtractor::FeatureExtract_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                               pcl::PointCloud<PointType>::Ptr& laserCloud,
                                               pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                               const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);
  }

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }
}

void LidarFeatureExtractor::FeatureExtract_Mid(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature){
    laserConerFeature->clear();
    laserSurfFeature->clear();
    for(auto & ptr : vlines){
        ptr->clear();
    }
    for(auto & v : vcorner){
        v.clear();
    }
    for(auto & v : vsurf){
        v.clear();
    }
    int cloud_num= msg->points.size();
    for(int i=0; i<cloud_num; ++i){
        int line_idx = std::round(msg->points[i].normal_y);
        msg->points[i].normal_z = _int_as_float(i);

        vlines[line_idx]->push_back(msg->points[i]);

        msg->points[i].normal_z = 0;
    }
    std::thread threads[N_SCANS];
    for(int i=0; i<N_SCANS; ++i){
        threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                                 std::ref(vcorner[i]), std::ref(vsurf[i]));
    }
    for(int i=0; i<N_SCANS; ++i){
        threads[i].join();
    }
    for(int i=0; i<N_SCANS; ++i){
        for(int j=0; j<vcorner[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
        }
        for(int j=0; j<vsurf[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
        }
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 1.0) < 1e-5)
            laserConerFeature->push_back(p);
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 2.0) < 1e-5)
            laserSurfFeature->push_back(p);
    }
}