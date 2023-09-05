#include "DbscanMethod.h"

double CalculateDistance(DbscanPoint* point1, DbscanPoint* point2)
{
	if (point1->xn.size() != point2->xn.size())
	{
		std::cout << "数据点异常"<< std::endl;
		return 0;
	}
	double distance = 0;
		
	double sum = 0;
	for (size_t i = 0; i != point1->xn.size(); ++i)
		sum += (point1->xn[i] - point2->xn[i]) * (point1->xn[i] - point2->xn[i]);

	distance = sqrt(sum);

	return distance;
}

void Dbscan(DbscanPoint* dbscan_point_ptr, int cluster_current)
{
	if (dbscan_point_ptr->visited == VISIT_NO)
	{
		S_temp++;
        std::cout << "访问了" << S_temp << "个点" << std::endl;

		dbscan_point_ptr->cluster += cluster_current;
		dbscan_point_ptr->visited = VISIT_YES;
	}
	else
		return;
	if (dbscan_point_ptr->point_type == POINT_CORE)
		for (auto m : dbscan_point_ptr->vec_dbscan_point_ptr)
			Dbscan(m, cluster_current);
	else
		dbscan_point_ptr->point_type = POINT_BOUNDARY;//本来这个点是噪声，但是这个点符合边界点的定义，所以改成边界点
}


void DbscanMethod(std::vector<DbscanPoint>& vec_dbscan_point, double Eps, int MinPts)
{
	size_t num = vec_dbscan_point.size();

    //第1步
    //计算所有两点间的距离
    //确定每个点的类型
    //存每个点后面的邻域内的点
	for (size_t i = 0; i != num; ++i)
	{
		for (size_t j = i + 1; j != num; ++j)
		{
			double distance = CalculateDistance(&vec_dbscan_point[i], &vec_dbscan_point[j]);
			if (distance <= Eps)
			{
                //密度增加
				vec_dbscan_point[i].num_pts++;				
				vec_dbscan_point[j].num_pts++;
                //将点记录到自己的邻域内
				vec_dbscan_point[i].vec_dbscan_point_ptr.push_back(&vec_dbscan_point[j]);
				vec_dbscan_point[j].vec_dbscan_point_ptr.push_back(&vec_dbscan_point[i]);

			}
		}
		if (vec_dbscan_point[i].num_pts >= MinPts)			//密度大于一定值就是核心点
			vec_dbscan_point[i].point_type = POINT_CORE;		
		else
			vec_dbscan_point[i].point_type = POINT_NOISE;	//剩下的暂时记为噪声
	}

    //第2步
    //开始类聚
	int cluster_current = 0;
	for (size_t i = 0; i != num; ++i)
		if (vec_dbscan_point[i].visited == VISIT_NO && vec_dbscan_point[i].point_type == POINT_CORE)
		{
			cluster_current++;//进入一个新的簇
			vec_dbscan_point[i].cluster += cluster_current;
			vec_dbscan_point[i].visited = VISIT_YES;

			S_temp++;
            std::cout << "访问了" << S_temp << "个点" << std::endl;

			for (auto m : vec_dbscan_point[i].vec_dbscan_point_ptr)
				Dbscan(m, cluster_current);
		}
}