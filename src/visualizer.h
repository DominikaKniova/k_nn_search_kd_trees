#ifndef PROJECT_VISUALIZER_H
#define PROJECT_VISUALIZER_H

#include <vector>
#include <queue>

#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkChartXY.h>
#include <vtkTable.h>
#include <vtkPlot.h>
#include <vtkFloatArray.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkPen.h>
#include <vtkPoints.h>
#include <vtkLine.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkNamedColors.h>
#include <vtkRegularPolygonSource.h>
#include <vtkPolygon.h>
#include <vtkSphereSource.h>

#include "kd_node.h"
#include "point.h"
#include "types.h"

//method for visualizations

vtkSmartPointer<vtkActor> plot_points(std::vector<Point> & points, int dimensionality, int step, double * color);

vtkSmartPointer<vtkActor> plot_range_points(std::vector<Point *> & data, int dimensionality, int step, double *color);

vtkSmartPointer<vtkActor> plot_range_points(std::deque<Point *> & data, int dimensionality, int step, double *color);

vtkSmartPointer<vtkActor> plot_query(const VectorXf &query, int dimensionality);

vtkSmartPointer<vtkActor> plot_range_sph(const VectorXf &query, float rad, int dimensionality);

vtkSmartPointer<vtkActor> plot_range_rect(std::vector<float> & ranges, const VectorXf & query, int dimensionality);

vtkSmartPointer<vtkActor> plot_visited_leaves(std::vector<std::vector<float>> & leaves, int dimensionality);

vtkSmartPointer<vtkActor> plot_splitting_lines(std::deque<std::vector<float>> & splitting_lines, int dimensionality, double * color);

vtkSmartPointer<vtkActor> plot_bounding_box(std::vector<float> & box, int dimensionality, double * color);

void show_visualization(std::vector<vtkSmartPointer<vtkActor>> & actors);

#endif //PROJECT_VISUALIZER_H
